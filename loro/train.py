"""
TRM (Tiny Recursive Model) for cognitive context assembly.
Autoresearch target: this is the ONLY file the agent modifies.

The TRM takes query embeddings and candidate embeddings,
iteratively refines a latent state, and outputs salience scores.
The eigenform is whatever the iterative process converges on.

Usage: uv run train.py
"""

import os
import gc
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    EMBED_DIM, CANDIDATE_POOL_SIZE, TOP_K, TIME_BUDGET,
    load_data, make_dataloader, evaluate_ndcg,
)

# ---------------------------------------------------------------------------
# TRM Model — The Eigenbehavior
# ---------------------------------------------------------------------------

# Architecture hyperparameters (autoresearch agent: modify these freely)
LATENT_DIM = 384            # dimension of the latent state z (match embed_dim)
N_ITERATIONS = 3            # K: number of recursive refinement steps
N_HEADS = 16                # attention heads in cross-attention
DROPOUT = 0.05              # dropout rate
USE_LAYER_NORM = True       # normalize between iterations

# Training hyperparameters
BATCH_SIZE = 32             # training batch size
LEARNING_RATE = 3e-3        # optimizer LR
WEIGHT_DECAY = 1e-2         # AdamW weight decay
WARMUP_STEPS = 100          # LR warmup steps
LABEL_SMOOTHING = 0.0       # smoothing for BCE loss
TRAIN_SECONDS = None        # override TIME_BUDGET (None = use prepare.py default, range: 30-120)


def load_judge_data(*args, **kwargs):
    """TODO: implement judge-data loader used by finetune_judge.py.

    Stub exists so `from train import load_judge_data` succeeds at import time.
    """
    raise NotImplementedError("load_judge_data is not implemented in train.py yet")


class TRM(nn.Module):
    """
    Tiny Recursive Model for context assembly.

    The eigenbehavior: z ← f(z, candidates, query) repeated K times.
    The eigenform: whatever salience scores converge to.

    Each iteration:
    1. Condition the latent on the query
    2. Cross-attend over candidates
    3. Score candidates from the refined latent
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        latent_dim: int = LATENT_DIM,
        n_iterations: int = N_ITERATIONS,
        n_heads: int = N_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.n_iterations = n_iterations
        self.latent_dim = latent_dim

        # Project embeddings to latent space
        self.query_proj = nn.Linear(embed_dim, latent_dim)
        self.cand_proj = nn.Linear(embed_dim, latent_dim)
        # Identity init: start in the pre-trained embedding space (since embed_dim == latent_dim)
        if embed_dim == latent_dim:
            nn.init.eye_(self.query_proj.weight)
            nn.init.zeros_(self.query_proj.bias)
            nn.init.eye_(self.cand_proj.weight)
            nn.init.zeros_(self.cand_proj.bias)

        # Cross-attention: latent attends to candidates (SHARED across iterations)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention among candidates (SHARED across iterations)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Latent update MLP (SHARED across iterations — true eigenbehavior)
        self.update_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )

        # Query re-injection gate (learned per-iteration mixing)
        self.query_gate = nn.Linear(latent_dim * 2, latent_dim)

        # Normalization
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)

        # Scoring head: project latent back to scalar per candidate
        self.score_head = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )

    def _iterate(self, z, c, q, res_scale):
        """Single iteration of the eigenbehavior. Factored out for reuse."""
        B, N, _ = z.shape

        # Self-attention: candidates attend to each other (compositional coherence)
        z_normed = self.norm1(z)
        self_out, _ = self.self_attn(z_normed, z_normed, z_normed)
        z = z + res_scale * self_out

        # Cross-attention: z attends to original candidates
        z_normed2 = self.norm2(z)
        cross_out, _ = self.cross_attn(z_normed2, c, c)
        z = z + res_scale * cross_out

        # Re-inject query signal (prevents drift from conditioning)
        q_expanded = q.unsqueeze(1).expand(-1, N, -1)
        gate = torch.sigmoid(self.query_gate(torch.cat([z, q_expanded], dim=-1)))
        z = z + res_scale * gate * q_expanded

        # MLP update
        z = z + res_scale * self.update_mlp(self.norm3(z))

        return z

    def _score(self, z, z_init, c):
        """Score from latent state. Factored out for reuse."""
        z = z + z_init
        z = z - z.mean(dim=1, keepdim=True)
        combined = torch.cat([z, c], dim=-1)
        return self.score_head(combined).squeeze(-1)

    def forward(self, query_emb: torch.Tensor, candidate_embs: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass with fixed K iterations.
        Used during training.

        Args:
            query_emb: (B, embed_dim) — the perturbation/conditioning
            candidate_embs: (B, N, embed_dim) — the candidate pool

        Returns:
            scores: (B, N) — salience scores for each candidate
        """
        B, N, _ = candidate_embs.shape

        # Project to latent space
        q = self.query_proj(query_emb)          # (B, latent_dim)
        c = self.cand_proj(candidate_embs)      # (B, N, latent_dim)

        # Dropout on both query and candidate projections — regularize both sides
        if self.training:
            q = F.dropout(q, p=0.1, training=True)
            c = F.dropout(c, p=0.05, training=True)

        # Initialize latent: element-wise product of query and candidates
        z = q.unsqueeze(1) * c                   # (B, N, latent_dim)
        z_init = z                                # save for skip connection

        # Iterative refinement — the eigenbehavior
        res_scale = 1.0 / math.sqrt(4.0)
        for k in range(self.n_iterations):
            z = self._iterate(z, c, q, res_scale)

        return self._score(z, z_init, c)

    @torch.no_grad()
    def forward_adaptive(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        max_k: int = 8,
        convergence_threshold: float = 0.01,
    ) -> dict:
        """
        Adaptive forward pass with convergence detection.
        The eigenform tells you when it's done by stopping changing.

        Inspired by SRLM (Apple, 2026): uncertainty-aware self-reflection.
        Instead of fixed K, runs until score delta drops below threshold.

        Returns dict with:
            scores: (B, N) final salience scores
            k_used: actual iterations taken
            converged: whether convergence was detected
            uncertainty: intrinsic uncertainty signals (SRLM-inspired)
        """
        B, N, _ = candidate_embs.shape

        q = self.query_proj(query_emb)
        c = self.cand_proj(candidate_embs)
        z = q.unsqueeze(1) * c
        z_init = z

        res_scale = 1.0 / math.sqrt(4.0)
        prev_scores = None
        k_used = 0

        for k in range(max_k):
            z = self._iterate(z, c, q, res_scale)
            k_used = k + 1

            # Score at this iteration
            current_scores = self._score(z, z_init, c)

            # Check convergence: has the ranking stabilized?
            if prev_scores is not None:
                # Score delta: mean absolute change in scores
                delta = (current_scores - prev_scores).abs().mean().item()
                # Rank stability: do the top-K agree?
                _, prev_top = prev_scores.topk(min(10, N), dim=-1)
                _, curr_top = current_scores.topk(min(10, N), dim=-1)
                rank_overlap = sum(
                    len(set(prev_top[b].tolist()) & set(curr_top[b].tolist()))
                    for b in range(B)
                ) / (B * min(10, N))

                if delta < convergence_threshold and rank_overlap > 0.9:
                    break

            prev_scores = current_scores

        # Compute intrinsic uncertainty signals (SRLM-inspired)
        scores = current_scores
        sorted_scores, _ = scores.sort(dim=-1, descending=True)

        # 1. Score margin: gap between top-1 and top-2 (decisiveness)
        if N >= 2:
            margin = (sorted_scores[:, 0] - sorted_scores[:, 1]).mean().item()
        else:
            margin = 0.0

        # 2. Score entropy: how spread out are the scores? (uncertainty)
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()

        # 3. Convergence speed: k_used / max_k (behavioral signal, like trace length)
        convergence_speed = k_used / max_k

        return {
            "scores": scores,
            "k_used": k_used,
            "converged": k_used < max_k,
            "uncertainty": {
                "margin": margin,           # high = confident (decisive separation)
                "entropy": entropy,          # low = confident (peaked distribution)
                "convergence_speed": convergence_speed,  # low = confident (converged fast)
            },
        }

    @torch.no_grad()
    def forward_speculative(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        n_trajectories: int = 8,
        max_k: int = 8,
        convergence_threshold: float = 0.01,
        temperature: float = 0.0,
    ) -> dict:
        """
        Speculative multi-trajectory inference.
        Runs N parallel trajectories with varied conditioning, selects the best
        using intrinsic uncertainty signals (SRLM joint selection).

        This runs in the inference gap — ~1ms per trajectory on Apple Silicon.
        N=8 trajectories = ~8ms total, while the LLM takes 2-10 seconds.

        Args:
            query_emb: (1, embed_dim) — single query
            candidate_embs: (1, N, embed_dim) — candidate pool
            n_trajectories: number of parallel hypotheses
            max_k: maximum iterations per trajectory
            convergence_threshold: convergence detection threshold
            temperature: noise level for trajectory diversity (0 = deterministic)

        Returns dict with:
            scores: (1, N) best trajectory's scores
            all_results: list of per-trajectory results
            selected_idx: which trajectory was selected
            self_consistency: fraction of trajectories agreeing on top-K
        """
        assert query_emb.size(0) == 1, "Speculative mode operates on single queries"

        results = []
        for t in range(n_trajectories):
            # Add trajectory-specific noise for diversity
            if temperature > 0 and t > 0:
                q_noise = query_emb + temperature * torch.randn_like(query_emb)
                q_noise = F.normalize(q_noise, p=2, dim=-1) * query_emb.norm(dim=-1, keepdim=True)
            else:
                q_noise = query_emb

            result = self.forward_adaptive(
                q_noise, candidate_embs,
                max_k=max_k,
                convergence_threshold=convergence_threshold,
            )
            results.append(result)

        # Self-consistency: do trajectories agree on top-K?
        top_k_sets = []
        for r in results:
            _, topk = r["scores"].topk(min(10, candidate_embs.size(1)), dim=-1)
            top_k_sets.append(set(topk[0].tolist()))

        # Find the plurality top-K (most common selections)
        from collections import Counter
        all_selected = Counter()
        for s in top_k_sets:
            all_selected.update(s)
        consensus_set = set(item for item, count in all_selected.most_common(10))

        # Self-consistency score: average overlap with consensus
        consistency = sum(
            len(s & consensus_set) / max(len(consensus_set), 1)
            for s in top_k_sets
        ) / n_trajectories

        # SRLM joint selection: pick trajectory with best uncertainty profile
        # s(p) = VC(p) * Len(p) — penalize low confidence AND slow convergence
        best_idx = 0
        best_score = float('-inf')
        for i, r in enumerate(results):
            u = r["uncertainty"]
            # Higher margin = more confident, lower entropy = more confident,
            # lower convergence_speed = faster convergence
            # Joint: high margin, low entropy, fast convergence
            joint = u["margin"] - u["entropy"] - u["convergence_speed"]
            if joint > best_score:
                best_score = joint
                best_idx = i

        return {
            "scores": results[best_idx]["scores"],
            "all_results": results,
            "selected_idx": best_idx,
            "self_consistency": consistency,
            "n_trajectories": n_trajectories,
            "best_uncertainty": results[best_idx]["uncertainty"],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_judge_ndcg(*args, **kwargs):
    """TODO: implement judge-set NDCG evaluation used by finetune_judge.py.

    Stub exists so `from train import evaluate_judge_ndcg` succeeds at import time.
    """
    raise NotImplementedError("evaluate_judge_ndcg is not implemented in train.py yet")


def main():
    parser = argparse.ArgumentParser(description="Train the TRM model")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Train for fixed step count (overrides time budget)")
    args = parser.parse_args()

    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    # Load data
    data = load_data()
    print(f"Loaded {data['n_queries']} queries, pool_size={data['pool_size']}, embed_dim={data['embed_dim']}")

    # Build model
    model = TRM(
        embed_dim=data["embed_dim"],
        latent_dim=LATENT_DIM,
        n_iterations=N_ITERATIONS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    ).to(device)

    num_params = count_parameters(model)
    print(f"TRM parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )

    # Data — synthetic + signal-weighted pairs + judge labels
    has_weights = "weights" in data
    if has_weights:
        print(f"Sample weights available: {data['weights'].shape[0]} entries, "
              f"mean={data['weights'].mean():.2f}, max={data['weights'].max():.2f}")
    train_loader = make_dataloader(data, BATCH_SIZE, "train")

    # Load judge data (mined from real sessions + retrospective + A/B comparisons)
    judge_loader = None
    judge_data_path = os.path.join(os.path.dirname(__file__), "judge_data.pt")
    if os.path.exists(judge_data_path):
        judge_labels = torch.load(judge_data_path, map_location="cpu", weights_only=False)
        if judge_labels:
            print(f"Judge data: {len(judge_labels)} labels (session-mined + retrospective + A/B)")
            # Pad/truncate to CANDIDATE_POOL_SIZE for uniform batching
            pool_size = CANDIDATE_POOL_SIZE
            embed_dim = EMBED_DIM
            padded_q, padded_c, padded_l = [], [], []
            for j in judge_labels:
                q_emb = j["query_emb"]
                c_emb = j["cand_embs"]
                labels = j["labels"]
                n = c_emb.size(0)
                if n >= pool_size:
                    padded_c.append(c_emb[:pool_size])
                    padded_l.append(labels[:pool_size])
                else:
                    pad_c = torch.zeros(pool_size - n, embed_dim)
                    pad_l = torch.zeros(pool_size - n)
                    padded_c.append(torch.cat([c_emb, pad_c], dim=0))
                    padded_l.append(torch.cat([labels, pad_l], dim=0))
                padded_q.append(q_emb)
            judge_q = torch.stack(padded_q)
            judge_c = torch.stack(padded_c)
            judge_l = torch.stack(padded_l)

            def _judge_iter():
                epoch = 1
                while True:
                    perm = torch.randperm(len(judge_q)).tolist()
                    for i in range(0, len(perm), BATCH_SIZE):
                        batch_idx = perm[i:i+BATCH_SIZE]
                        yield judge_q[batch_idx], judge_c[batch_idx], judge_l[batch_idx], epoch
                    epoch += 1
            judge_loader = _judge_iter()

    # Training loop — time budget
    time_budget = TRAIN_SECONDS if TRAIN_SECONDS is not None else TIME_BUDGET
    time_budget = max(30, min(120, time_budget))
    print(f"Time budget: {time_budget}s")
    if args.max_steps:
        print(f"Max steps: {args.max_steps} (overrides time budget)")
    print(f"Training...")

    model.train()
    step = 0
    total_training_time = 0
    smooth_loss = 0
    best_ndcg = 0

    use_judge = False

    while True:
        t0 = time.time()

        # Alternate: every other step uses judge data (if available)
        use_judge = not use_judge
        w_batch = None
        if use_judge and judge_loader is not None:
            q_batch, c_batch, l_batch, epoch = next(judge_loader)
        else:
            batch_data = next(train_loader)
            if len(batch_data) == 5:
                q_batch, c_batch, l_batch, epoch, w_batch = batch_data
            else:
                q_batch, c_batch, l_batch, epoch = batch_data

        q_batch = q_batch.to(device)
        c_batch = c_batch.to(device)
        l_batch = l_batch.to(device)
        if w_batch is not None:
            w_batch = w_batch.to(device)

        # Mixup augmentation: interpolate between pairs of examples
        if q_batch.size(0) > 1:
            lam = torch.distributions.Beta(0.2, 0.2).sample().item()
            perm = torch.randperm(q_batch.size(0), device=device)
            q_batch = lam * q_batch + (1 - lam) * q_batch[perm]
            c_batch = lam * c_batch + (1 - lam) * c_batch[perm]
            l_batch = lam * l_batch + (1 - lam) * l_batch[perm]
            if w_batch is not None:
                w_batch = lam * w_batch + (1 - lam) * w_batch[perm]

        # Forward
        scores = model(q_batch, c_batch)

        # Loss: BCE with logits (binary relevance labels)
        # Apply per-sample signal weights if available
        targets_smooth = l_batch * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        if w_batch is not None:
            # Per-sample weighted loss: higher weight for stronger signals
            per_sample_loss = F.binary_cross_entropy_with_logits(
                scores, targets_smooth, reduction='none'
            ).mean(dim=-1)  # (B,) — mean over candidates per sample
            loss = (per_sample_loss * w_batch).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(scores, targets_smooth)
        if use_judge and judge_loader is not None:
            loss = loss * 1.0  # equal weight — judge data tripled, no amplification needed

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # LR schedule: linear warmup then cosine decay
        if step < WARMUP_STEPS:
            lr_mult = (step + 1) / WARMUP_STEPS
        else:
            progress = min(total_training_time / time_budget, 1.0)
            lr_mult = 0.5 * (1 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = LEARNING_RATE * lr_mult

        optimizer.step()

        dt = time.time() - t0
        if step > 5:
            total_training_time += dt

        loss_val = loss.item()
        ema = 0.95
        smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
        debiased = smooth_loss / (1 - ema ** (step + 1))

        if step % 50 == 0:
            progress = min(total_training_time / time_budget, 1.0) * 100
            remaining = max(0, time_budget - total_training_time)
            print(f"\rstep {step:05d} ({progress:.1f}%) | loss: {debiased:.6f} | lr: {lr_mult:.4f} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # GC management
        if step == 0:
            gc.collect()

        step += 1

        if args.max_steps and step >= args.max_steps:
            break
        if step > 5 and total_training_time >= time_budget:
            break

    print()

    # Final evaluation
    print("Evaluating...")
    model.eval()
    val_ndcg = evaluate_ndcg(model, data, batch_size=BATCH_SIZE, device=str(device))

    # Summary
    t_end = time.time()
    print("---")
    print(f"val_ndcg:         {val_ndcg:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params:,}")
    print(f"latent_dim:       {LATENT_DIM}")
    print(f"n_iterations:     {N_ITERATIONS}")
    print(f"n_heads:          {N_HEADS}")


if __name__ == "__main__":
    main()
