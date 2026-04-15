#!/usr/bin/env python3
"""
LoRO Mamba TRM — Temporal Retrieval Model using Selective State Spaces.

The TRM processes temporally ordered session events (queries + retrievals)
through a Mamba SSM, maintaining a hidden state (the "light cone") that
compresses the observer's trajectory through the workspace. It predicts
which chunks will be retrieved next given the trajectory so far.

Architecture:
    Event embeddings → [Type embed + Linear] → [Mamba blocks] → [Score head] → scores

Training objective:
    Next-retrieval prediction: given prefix events, score candidate chunks
    for the next retrieval step.

Training modes:
    Step-based (default): Trains for exactly TRAIN_STEPS (3200) steps.
        Reproducible across runs regardless of hardware speed.
        Override with --max-steps N.
    Time-based (deprecated): Trains for a wall-clock budget in seconds.
        Use --time-budget S to enable. Not reproducible across machines.

Inference:
    step() method processes one event at a time, maintaining hidden state
    for microsecond-latency proprioceptive speculation.
"""

import argparse
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# ── Hyperparameters (Ralph tunes these) ──────────────────────────────

D_MODEL = 384       # Must match embedding dim
D_STATE = 4         # SSM state dimension (light cone capacity)
D_CONV = 2          # Local convolution width
N_LAYERS = 2        # Number of Mamba blocks
EXPAND_FACTOR = 1   # Inner dimension = D_MODEL * EXPAND_FACTOR
N_EVENT_TYPES = 4   # query, retrieval, search, edit
DROPOUT = 0.05
CANDIDATE_POOL = 64
TOP_K = 10

LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 50
BATCH_SIZE = 16
TRAIN_STEPS = 3200  # Step-based: prevent overtraining on fast machines
GRAD_CLIP = 0.5
EVAL_INTERVAL = 100  # Evaluate val NDCG every N steps, keep best checkpoint
SEED = 42
INFONCE_TAU = 0.02    # InfoNCE temperature (LRAT Section 5.2.2)

# ── Paths ────────────────────────────────────────────────────────────

CACHE_DIR = Path(os.path.expanduser("~/.cache/cogos-autoresearch"))
SEQUENCES_PATH = CACHE_DIR / "sequences.pt"
BEST_MODEL_PATH = Path("best_model_mamba.pt")
RESULTS_PATH = Path("results_mamba.tsv")

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ── Stochastic Depth (DropPath) ──────────────────────────────────────

class DropPath(nn.Module):
    """Drop entire residual path with probability drop_prob during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x / keep_prob * rand


# ── Selective SSM (Mamba-style) ──────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Simplified Mamba selective state space block.

    Uses input-dependent B, C, and delta (step size) to achieve
    content-aware state updates. The state vector is the compressed
    light cone — the observer's trajectory through state space.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        # Input projection: x → (z, x_ssm) for gating
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 1D conv for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, padding=d_conv - 1,
            groups=d_inner, bias=True,
        )

        # SSM parameters — input-dependent (selective)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta

        # Learnable log(A) — initialized for HiPPO-like decay
        log_A = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.log_A = nn.Parameter(log_A.unsqueeze(0).expand(d_inner, -1).clone())

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.norm = nn.LayerNorm(d_model)

        self.d_inner = d_inner

    def forward(self, x: torch.Tensor, init_state: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence
            init_state: (B, d_inner, d_state) optional initial SSM state
        Returns:
            (B, L, D) output sequence
        """
        residual = x
        x = self.norm(x)
        B, L, D = x.shape

        # Project and split into SSM input and gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1D conv for local context
        x_ssm = x_ssm.transpose(1, 2)  # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]  # causal: trim to L
        x_ssm = x_ssm.transpose(1, 2)  # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # Input-dependent SSM parameters (selective mechanism)
        x_params = self.x_proj(x_ssm)  # (B, L, d_state*2 + 1)
        B_param = x_params[:, :, :self.d_state]
        C_param = x_params[:, :, self.d_state:2*self.d_state]
        delta = F.softplus(x_params[:, :, -1:])  # step size = aperture

        # Discretize A
        A = -torch.exp(self.log_A)  # (d_inner, d_state), negative for stability
        dA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Sequential scan (with optional initial state)
        y = self._sequential_scan(x_ssm, dA, dB, C_param, init_state)

        # Gate and project
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + self.drop_path(y)

    def _sequential_scan(self, x, dA, dB, C, init_state=None):
        """Sequential scan implementation."""
        B_batch, L, d_inner = x.shape
        h = (init_state if init_state is not None
             else torch.zeros(B_batch, d_inner, self.d_state, device=x.device, dtype=x.dtype))
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]
            ys.append(y_t)
        return torch.stack(ys, dim=1)

    def step(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """Single-step inference for proprioceptive speculation."""
        B_batch = x.shape[0]
        x_norm = self.norm(x.unsqueeze(1)).squeeze(1)

        xz = self.in_proj(x_norm)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm = F.silu(x_ssm)

        x_params = self.x_proj(x_ssm)
        B_param = x_params[:, :self.d_state]
        C_param = x_params[:, self.d_state:2*self.d_state]
        delta = F.softplus(x_params[:, -1:])

        A = -torch.exp(self.log_A)
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(1)

        if state is None:
            state = torch.zeros(B_batch, self.d_inner, self.d_state,
                                device=x.device, dtype=x.dtype)

        new_state = dA * state + dB * x_ssm.unsqueeze(-1)
        y = (new_state * C_param.unsqueeze(1)).sum(-1) + self.D * x_ssm

        y = y * F.silu(z)
        y = self.out_proj(y)
        return y, new_state


# ── Attention Probe ──────────────────────────────────────────────────

class AttentionProbe(nn.Module):
    """
    Single-head attention probe: lets trajectory context attend over candidate set.
    The context can see all candidates before scoring, enabling comparative ranking.
    """
    def __init__(self, d_model: int, d_head: int = 64, attn_dropout: float = 0.0):
        super().__init__()
        self.d_head = d_head
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.out_proj = nn.Linear(d_head, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

    def forward(self, context: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, D) trajectory context from Mamba
            candidates: (B, N, D) candidate embeddings
        Returns:
            (B, D) attention-enriched context
        """
        Q = self.q_proj(context).unsqueeze(1)          # (B, 1, d_head)
        K = self.k_proj(candidates)                     # (B, N, d_head)
        V = self.v_proj(candidates)                     # (B, N, d_head)
        attn = F.softmax(
            (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head), dim=-1
        )                                               # (B, 1, N)
        attn = self.attn_drop(attn)
        out = (attn @ V).squeeze(1)                     # (B, d_head)
        return self.out_proj(out)                       # (B, D)


# ── Mamba TRM Model ─────────────────────────────────────────────────

class MambaTRM(nn.Module):
    """
    Temporal Retrieval Model using Mamba selective state spaces.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_state: int = D_STATE,
        d_conv: int = D_CONV,
        n_layers: int = N_LAYERS,
        expand: int = EXPAND_FACTOR,
        n_event_types: int = N_EVENT_TYPES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.type_embed = nn.Embedding(n_event_types, d_model)
        self.input_proj = nn.Linear(d_model * 2, d_model)

        self.layers = nn.ModuleList([
            SelectiveSSM(d_model, d_state, d_conv, expand, dropout, drop_path=0.05)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Four attention probes: iterative context refinement over candidates
        self.attn_probes = nn.ModuleList([
            AttentionProbe(d_model, d_head=128) for _ in range(4)
        ])
        # Pre-norm for each probe round (transformer-style)
        self.probe_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(4)
        ])

        self.score_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "log_A" in name:
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, events, event_types, candidates, prefix_lens):
        B, L, D = events.shape

        type_emb = self.type_embed(event_types)
        x = self.input_proj(torch.cat([events, type_emb], dim=-1))

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # Max+mean pooling over valid trajectory positions
        mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < prefix_lens.unsqueeze(1)
        x_masked_max = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        max_ctx = x_masked_max.max(dim=1)[0]
        mean_ctx = (x * mask.unsqueeze(-1).float()).sum(1) / prefix_lens.unsqueeze(1).float()
        context = max_ctx + mean_ctx

        # Iterative context refinement with pre-norm: attend over candidates N times
        for probe, norm in zip(self.attn_probes, self.probe_norms):
            context = context + probe(norm(context), candidates)

        context_exp = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        combined = torch.cat([context_exp, candidates], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        return scores

    def step(self, event, event_type, states=None):
        """Single-step inference for proprioceptive speculation."""
        type_emb = self.type_embed(event_type)
        x = self.input_proj(torch.cat([event, type_emb], dim=-1))

        if states is None:
            states = [None] * self.n_layers

        new_states = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer.step(x, state)
            new_states.append(new_state)

        x = self.final_norm(x)
        return x, new_states

    def score_candidates(self, context, candidates):
        """Score candidates against a trajectory context from step()."""
        for probe, norm in zip(self.attn_probes, self.probe_norms):
            context = context + probe(norm(context), candidates)
        context_exp = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        combined = torch.cat([context_exp, candidates], dim=-1)
        return self.score_head(combined).squeeze(-1)

    def get_light_cone(self, states):
        """Extract compressed light cone as a single vector."""
        stacked = torch.stack([s.mean(dim=-1).mean(dim=-1) for s in states])
        return stacked.mean(dim=0)


# ── Data Loading ─────────────────────────────────────────────────────

def load_data(seed: int = SEED):
    data = torch.load(SEQUENCES_PATH, weights_only=False)
    print(f"Loaded {len(data)} temporal sequences")

    n_val = max(1, len(data) // 5)
    torch.manual_seed(seed)
    perm = torch.randperm(len(data))
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data


def collate_batch(samples, device=DEVICE):
    max_len = max(s["prefix_len"] for s in samples)
    B = len(samples)
    D = samples[0]["prefix_events"].shape[-1]
    N = samples[0]["candidates"].shape[0]

    events = torch.zeros(B, max_len, D)
    types = torch.zeros(B, max_len, dtype=torch.long)
    candidates = torch.zeros(B, N, D)
    labels = torch.zeros(B, N)
    lens = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(samples):
        L = s["prefix_len"]
        events[i, :L] = s["prefix_events"][:L]
        types[i, :L] = s["prefix_types"][:L]
        candidates[i] = s["candidates"]
        labels[i] = s["labels"]
        lens[i] = L

    return (
        events.to(device), types.to(device), candidates.to(device),
        labels.to(device), lens.to(device),
    )


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_ndcg(scores, labels, k=TOP_K):
    B = scores.shape[0]
    ndcgs = []
    for i in range(B):
        s, l = scores[i], labels[i]
        _, top_idx = s.topk(min(k, s.shape[0]))
        dcg = (l[top_idx] / torch.log2(torch.arange(2, k + 2, device=s.device).float()[:len(top_idx)])).sum()
        _, ideal_idx = l.topk(min(k, l.shape[0]))
        idcg = (l[ideal_idx] / torch.log2(torch.arange(2, k + 2, device=s.device).float()[:len(ideal_idx)])).sum()
        ndcgs.append((dcg / idcg).item() if idcg > 0 else 0.0)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def cosine_baseline(val_data, device=DEVICE):
    ndcgs = []
    for s in val_data:
        query = s["prefix_events"][s["prefix_len"] - 1].unsqueeze(0).to(device)
        cands = s["candidates"].unsqueeze(0).to(device)
        labs = s["labels"].unsqueeze(0).to(device)
        scores = F.cosine_similarity(query.unsqueeze(1), cands, dim=-1)
        ndcgs.append(evaluate_ndcg(scores, labs))
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


# ── InfoNCE Contrastive Loss (LRAT Section 5.2.2, Eq. 4) ───────────

def infonce_loss(scores, labels, weights=None, tau=INFONCE_TAU,
                 query_embeddings=None, in_batch_pos_embeddings=None):
    """
    Weighted InfoNCE contrastive loss with optional in-batch negatives.

    For each query in the batch, treats the candidate with the highest
    relevance label as the positive and all others as in-pool negatives.
    The temperature tau sharpens the contrastive signal.

    When query_embeddings and in_batch_pos_embeddings are provided, other
    queries' positive documents become additional hard negatives for each
    query (cross-query negatives), giving B-1 extra negatives for free.

    Args:
        scores: (B, N) model output scores for each candidate
        labels: (B, N) relevance labels (binary or graded)
        weights: (B,) per-sample intensity weights (optional, from LRAT Eq.3)
        tau: temperature parameter (default 0.02, per LRAT)
        query_embeddings: (B, D) query embeddings for cross-query negatives (optional)
        in_batch_pos_embeddings: (B, D) positive candidate embeddings per query (optional)

    Returns:
        scalar loss
    """
    # Find positive index: candidate with highest label per row.
    # argmax returns first index on ties, satisfying the edge-case spec.
    pos_idx = labels.argmax(dim=1)                          # (B,)

    # Scale scores by temperature
    logits = scores / tau                                    # (B, N)

    # Gather positive logits: one per sample
    pos_logits = logits.gather(1, pos_idx.unsqueeze(1)).squeeze(1)  # (B,)

    # Log-sum-exp over all candidates (denominator)
    log_sum_exp = torch.logsumexp(logits, dim=1)             # (B,)

    # Incorporate in-batch cross-query negatives when provided
    if query_embeddings is not None and in_batch_pos_embeddings is not None:
        # Cross-query similarity: each query scored against all positives
        cross_scores = query_embeddings @ in_batch_pos_embeddings.T / tau  # (B, B)

        # Mask diagonal: a query's own positive is already in the in-pool set
        B = cross_scores.size(0)
        diag_mask = torch.eye(B, dtype=torch.bool, device=cross_scores.device)
        cross_scores = cross_scores.masked_fill(diag_mask, float('-inf'))  # exp(-inf) = 0

        # Combine in-pool and cross-query denominators via log-sum-exp
        # log(exp(lse_pool) + exp(lse_cross)) = log(sum_pool + sum_cross)
        cross_log_sum_exp = torch.logsumexp(cross_scores, dim=1)  # (B,)
        log_sum_exp = torch.logaddexp(log_sum_exp, cross_log_sum_exp)  # (B,)

    # Per-sample InfoNCE loss: -log(exp(s_pos/tau) / sum(exp(s_all/tau)))
    per_sample_loss = log_sum_exp - pos_logits               # (B,)

    # Apply per-sample intensity weights if provided
    if weights is not None:
        per_sample_loss = per_sample_loss * weights

    return per_sample_loss.mean()


def gather_positives(candidate_embeddings, labels):
    """Extract the positive candidate embedding for each query in the batch.

    Args:
        candidate_embeddings: (B, N, D) candidate embeddings
        labels: (B, N) relevance labels

    Returns:
        (B, D) positive embeddings (highest-label candidate per query)
    """
    pos_idx = labels.argmax(dim=1)  # (B,)
    return candidate_embeddings[torch.arange(len(pos_idx)), pos_idx]  # (B, D)


# ── Training Loop ────────────────────────────────────────────────────

def train(args=None):
    # Parse defaults if called without args (backward compat)
    if args is None:
        args = argparse.Namespace(max_steps=TRAIN_STEPS, time_budget=None, seed=SEED, warm_start=None, loss='infonce')

    # Resolve training termination mode
    use_time_budget = args.time_budget is not None
    max_steps = args.max_steps
    time_budget = args.time_budget
    seed = getattr(args, "seed", SEED)
    lr = LEARNING_RATE * 0.1 if args.warm_start else LEARNING_RATE

    if use_time_budget:
        print("WARNING: --time-budget is deprecated. Use --max-steps for reproducible training.")
        print(f"Training with time budget: {time_budget}s, seed: {seed}, lr: {lr:.2e}")
    else:
        print(f"Training with step budget: {max_steps} steps, seed: {seed}, lr: {lr:.2e}")

    loss_mode = getattr(args, 'loss', 'infonce')
    if loss_mode == 'infonce':
        print(f"Loss: infonce (tau={INFONCE_TAU})")
    else:
        print("Loss: ce (label smoothing)")

    torch.manual_seed(seed)
    if DEVICE == "mps":
        torch.mps.manual_seed(seed)

    train_data, val_data = load_data(seed=seed)

    cos_ndcg = cosine_baseline(val_data)
    print(f"Cosine baseline NDCG@{TOP_K}: {cos_ndcg:.6f}")

    model = MambaTRM(
        d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
        n_layers=N_LAYERS, expand=EXPAND_FACTOR, dropout=DROPOUT,
    ).to(DEVICE)

    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f'Warm-start: loaded weights from {args.warm_start} (NDCG: {checkpoint.get("ndcg", "unknown")})')
        args._checkpoint_ndcg = checkpoint.get("ndcg", 0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = AdamW(
        model.parameters(), lr=lr,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99),
    )

    best_ndcg = 0.0
    best_state = None
    ema_state = None
    EMA_DECAY = 0.999
    EMA_START = 1600
    step = 0
    start = time.time()

    def should_continue():
        if use_time_budget:
            return (time.time() - start) < time_budget
        return step < max_steps

    while should_continue():
        model.train()

        indices = torch.randint(0, len(train_data), (min(BATCH_SIZE, len(train_data)),))
        batch = [train_data[i] for i in indices]
        events, types, candidates, labels, lens = collate_batch(batch)
        # Input noise augmentation: small perturbation of SBERT embeddings
        events = events + torch.randn_like(events) * 0.01
        candidates = candidates + torch.randn_like(candidates) * 0.01

        scores = model(events, types, candidates, lens)

        if args.loss == 'infonce':
            # LRAT-style InfoNCE with optional in-batch negatives
            # Use last event embedding in each prefix as query proxy
            query_embs = events[:, -1, :]  # (B, D) last event embedding as query
            pos_embs = gather_positives(candidates, labels)  # (B, D)

            # Per-sample weights will come from the data loader in a future task
            batch_weights = None

            loss = infonce_loss(
                scores, labels,
                weights=batch_weights,
                tau=INFONCE_TAU,
                query_embeddings=query_embs,
                in_batch_pos_embeddings=pos_embs,
            )
        else:
            # Legacy cross-entropy with label smoothing
            step_frac = step / max_steps
            eps = 0.1 * max(0.0, 1.0 - step_frac)
            if eps > 0:
                smoothed = labels * (1 - eps) + eps / CANDIDATE_POOL
                label_probs = smoothed / smoothed.sum(dim=-1, keepdim=True)
            else:
                label_probs = labels / labels.sum(dim=-1, keepdim=True)
            log_probs = F.log_softmax(scores, dim=-1)
            loss = -(label_probs * log_probs).sum(-1).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step < WARMUP_STEPS:
            lr_scale = (step + 1) / WARMUP_STEPS
            for pg in optimizer.param_groups:
                pg["lr"] = lr * lr_scale

        step += 1

        # EMA: exponential moving average starting at 50% of training
        if step >= EMA_START:
            with torch.no_grad():
                if ema_state is None:
                    ema_state = {k: v.clone().float() for k, v in model.state_dict().items()}
                else:
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(EMA_DECAY).add_(v.float(), alpha=1 - EMA_DECAY)

        if step % 50 == 0:
            elapsed = time.time() - start
            if use_time_budget:
                print(f"  step {step:4d} | loss {loss.item():.4f} | "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                      f"{elapsed:.0f}s/{time_budget}s")
            else:
                print(f"  step {step:4d} | loss {loss.item():.4f} | "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                      f"{elapsed:.0f}s | {step}/{max_steps}")

        # Periodic val eval: track best checkpoint
        if step % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                all_scores, all_labels = [], []
                for s in val_data:
                    ev = s["prefix_events"][:s["prefix_len"]].unsqueeze(0).to(DEVICE)
                    ty = s["prefix_types"][:s["prefix_len"]].unsqueeze(0).to(DEVICE)
                    ca = s["candidates"].unsqueeze(0).to(DEVICE)
                    la = s["labels"].unsqueeze(0).to(DEVICE)
                    ln = torch.tensor([s["prefix_len"]], device=DEVICE)
                    sc = model(ev, ty, ca, ln)
                    all_scores.append(sc)
                    all_labels.append(la)
                val_ndcg = evaluate_ndcg(torch.cat(all_scores), torch.cat(all_labels))
            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # EMA: load EMA weights and compare to best checkpoint
    if ema_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in ema_state.items()})
        model.eval()
        with torch.no_grad():
            all_scores, all_labels = [], []
            for s in val_data:
                ev = s["prefix_events"][:s["prefix_len"]].unsqueeze(0).to(DEVICE)
                ty = s["prefix_types"][:s["prefix_len"]].unsqueeze(0).to(DEVICE)
                ca = s["candidates"].unsqueeze(0).to(DEVICE)
                la = s["labels"].unsqueeze(0).to(DEVICE)
                ln = torch.tensor([s["prefix_len"]], device=DEVICE)
                sc = model(ev, ty, ca, ln)
                all_scores.append(sc)
                all_labels.append(la)
            ema_ndcg = evaluate_ndcg(torch.cat(all_scores), torch.cat(all_labels))
        print(f"EMA NDCG: {ema_ndcg:.6f}, Best ckpt NDCG: {best_ndcg:.6f}")
        if ema_ndcg < best_ndcg and best_state is not None:
            model.load_state_dict(best_state)
            print("Using best checkpoint (better than EMA)")
        else:
            best_ndcg = ema_ndcg
            print("Using EMA weights")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_scores, all_labels = [], []
        for s in val_data:
            ev = s["prefix_events"][:s["prefix_len"]].unsqueeze(0).to(DEVICE)
            ty = s["prefix_types"][:s["prefix_len"]].unsqueeze(0).to(DEVICE)
            ca = s["candidates"].unsqueeze(0).to(DEVICE)
            la = s["labels"].unsqueeze(0).to(DEVICE)
            ln = torch.tensor([s["prefix_len"]], device=DEVICE)

            sc = model(ev, ty, ca, ln)
            all_scores.append(sc)
            all_labels.append(la)

        scores_cat = torch.cat(all_scores)
        labels_cat = torch.cat(all_labels)
        final_ndcg = evaluate_ndcg(scores_cat, labels_cat)

    print(f"\n{'='*60}")
    print(f"Final NDCG@{TOP_K}: {final_ndcg:.6f}")
    print(f"Cosine baseline:    {cos_ndcg:.6f}")
    print(f"Delta:              {(final_ndcg - cos_ndcg)*1000:+.1f} pts")
    print(f"Steps:              {step}")
    print(f"Params:             {n_params:,}")
    print(f"{'='*60}")

    # NDCG regression gate (warm-start only)
    checkpoint_ndcg = getattr(args, '_checkpoint_ndcg', None)
    if checkpoint_ndcg is not None:
        regression = checkpoint_ndcg - final_ndcg
        if regression > 0.02:
            print(f"\nREGRESSION GATE: FAIL")
            print(f"  Delta-trained NDCG ({final_ndcg:.6f}) regressed > 0.02 "
                  f"from checkpoint ({checkpoint_ndcg:.6f})")
            print(f"  Rejecting delta update. Previous model retained.")
            import sys
            sys.exit(2)
        else:
            print(f"\nREGRESSION GATE: PASS")
            print(f"  Delta-trained NDCG ({final_ndcg:.6f}) vs checkpoint "
                  f"({checkpoint_ndcg:.6f}): delta {(final_ndcg - checkpoint_ndcg)*1000:+.1f} pts")

    if final_ndcg >= best_ndcg:
        torch.save({
            "model_state_dict": model.state_dict(),
            "ndcg": final_ndcg,
            "cosine_baseline": cos_ndcg,
            "n_params": n_params,
            "config": {
                "d_model": D_MODEL, "d_state": D_STATE, "d_conv": D_CONV,
                "n_layers": N_LAYERS, "expand": EXPAND_FACTOR, "dropout": DROPOUT,
            },
        }, BEST_MODEL_PATH)
        print(f"Saved best model to {BEST_MODEL_PATH}")

    # Append to results
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text("commit\tndcg\tparams\tstatus\tdescription\n")

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    with open(RESULTS_PATH, "a") as f:
        status = "keep" if final_ndcg > cos_ndcg else "discard"
        desc = (f"Mamba d_state={D_STATE} n_layers={N_LAYERS} "
                f"expand={EXPAND_FACTOR} lr={LEARNING_RATE}")
        f.write(f"{commit}\t{final_ndcg:.6f}\t{n_params}\t{status}\t{desc}\n")

    return final_ndcg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRO Mamba TRM (step-based by default for reproducibility)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=TRAIN_STEPS,
        help=f"Maximum training steps (default: {TRAIN_STEPS}). "
             "Ignored if --time-budget is set.",
    )
    parser.add_argument(
        "--time-budget", type=float, default=None,
        help="DEPRECATED. Wall-clock training budget in seconds. "
             "Enables time-based training instead of step-based. "
             "Not reproducible across machines.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed for reproducibility (default: {SEED}).",
    )
    parser.add_argument(
        '--warm-start', type=str, default=None,
        help='Path to previous best model checkpoint for warm-start training. Uses 0.1x learning rate.',
    )
    parser.add_argument('--loss', choices=['infonce', 'ce'], default='infonce',
        help='Loss function: infonce (LRAT-style, default) or ce (cross-entropy with label smoothing)')
    return parser.parse_args()


if __name__ == "__main__":
    # ── InfoNCE smoke test ──────────────────────────────────────────
    if os.environ.get("INFONCE_SMOKE_TEST"):
        torch.manual_seed(SEED)
        B, N = 4, CANDIDATE_POOL
        fake_scores = torch.randn(B, N)
        fake_labels = torch.zeros(B, N)
        fake_labels[torch.arange(B), torch.randint(0, N, (B,))] = 1.0

        # Basic call — must return a finite scalar
        loss = infonce_loss(fake_scores, fake_labels)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss.item() > 0, f"InfoNCE loss should be positive, got {loss.item()}"

        # Weighted call — must also return a finite scalar
        fake_weights = torch.rand(B)
        wloss = infonce_loss(fake_scores, fake_labels, weights=fake_weights)
        assert wloss.dim() == 0, f"Expected scalar, got shape {wloss.shape}"
        assert torch.isfinite(wloss), f"Weighted loss is not finite: {wloss}"

        # Perfect scores should yield near-zero loss
        perfect_scores = fake_labels * 100.0
        perfect_loss = infonce_loss(perfect_scores, fake_labels)
        assert perfect_loss.item() < 0.1, f"Perfect-score loss too high: {perfect_loss.item()}"

        print(f"InfoNCE smoke test PASSED (loss={loss.item():.4f}, "
              f"weighted={wloss.item():.4f}, perfect={perfect_loss.item():.6f})")
    else:
        train(parse_args())
