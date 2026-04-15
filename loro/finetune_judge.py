"""
Fine-tune TRM from best_model.pt using ONLY judge-win examples.

Key insight from eval analysis:
  - 21/26 judge verdicts preferred cosine — training on those labels would
    teach TRM to behave like cosine, destroying its learned doc-coherence signal.
  - Only the 5 TRM-win examples are useful: they show where document-level
    coherence actually beats surface similarity.
  - Cosine-win labels = actively harmful (contradict the synthetic training).

Strategy:
  1. Load best_model.pt (good synthetic checkpoint, NDCG ~0.802)
  2. Filter judge_data.pt to TRM-win examples only
  3. Fine-tune at low LR with 1:5 judge:synthetic interleaving
  4. Save as judge_model.pt — evaluate on held-out queries

Usage:
    uv run finetune_judge.py [--time 30] [--lr 3e-4]
"""

import os
import math
import time
import argparse

import torch
import torch.nn.functional as F

from prepare import (
    EMBED_DIM, CANDIDATE_POOL_SIZE, TOP_K, TIME_BUDGET,
    load_data, make_dataloader, evaluate_ndcg,
)
from train import TRM, load_judge_data, evaluate_judge_ndcg

JUDGE_DATA_PATH = os.path.join(os.path.dirname(__file__), "judge_data.pt")
BEST_CKPT = os.path.join(os.path.dirname(__file__), "best_model.pt")
JUDGE_CKPT = os.path.join(os.path.dirname(__file__), "judge_model.pt")


def filter_trm_wins(judge_data_path: str) -> list[dict]:
    """
    Load judge_data.pt and return only TRM-win examples.
    Drops cosine-win and tie examples.
    """
    examples = torch.load(judge_data_path, map_location="cpu", weights_only=False)
    trm_wins = [ex for ex in examples if ex.get("winner") == "trm"]
    # Deduplicate by query_text (the file has 10x reps)
    seen = set()
    unique_wins = []
    for ex in trm_wins:
        qt = ex.get("query_text", "")
        if qt not in seen:
            seen.add(qt)
            unique_wins.append(ex)
    print(f"Loaded {len(examples)} judge examples → {len(trm_wins)} TRM-wins "
          f"({len(unique_wins)} unique queries)")
    return trm_wins   # Return all reps for upweighting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=30.0,
                        help="Fine-tuning time budget in seconds")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Fine-tuning learning rate (much lower than pre-training)")
    parser.add_argument("--syn-ratio", type=int, default=5,
                        help="Synthetic batches per 1 judge batch (default: 5:1 syn:judge)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Fine-tuning batch size (smaller for tiny judge set)")
    parser.add_argument("--judge-data", type=str, default=JUDGE_DATA_PATH,
                        help="Path to judge_data.pt (default: judge_data.pt)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load base model
    assert os.path.exists(BEST_CKPT), f"No checkpoint at {BEST_CKPT}"
    ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    base_ndcg = ckpt.get("val_ndcg", "?")
    print(f"Loaded base model: NDCG={base_ndcg:.4f}")

    model = TRM(
        embed_dim=cfg["embed_dim"],
        latent_dim=cfg["latent_dim"],
        n_iterations=cfg["n_iterations"],
        n_heads=cfg["n_heads"],
        dropout=cfg.get("dropout", 0.05),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load synthetic data (for regularization during fine-tuning)
    syn_data = load_data()
    syn_loader = make_dataloader(syn_data, args.batch_size, "train")
    print(f"Synthetic data: {syn_data['n_queries']} queries (for regularization)")

    # Load TRM-win judge examples only
    judge_data_path = args.judge_data
    assert os.path.exists(judge_data_path), f"No judge data at {judge_data_path}"
    trm_win_examples = filter_trm_wins(judge_data_path)

    # Pad to fixed pool size
    judge_data = _pad_examples(trm_win_examples)
    judge_loader = make_dataloader(judge_data, args.batch_size, "train")
    print(f"TRM-win fine-tune examples: {judge_data['n_queries']} "
          f"(unique: {judge_data['n_queries'] // 10})")
    print(f"Mix: 1 judge per {args.syn_ratio} synthetic | LR={args.lr} | budget={args.time}s")

    # Optimizer — lower LR, less weight decay to preserve learned weights
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,   # reduced from 1e-2
    )

    print("Fine-tuning...")
    model.train()
    step = 0
    total_time = 0
    smooth_loss = 0

    while True:
        t0 = time.time()

        # 1 judge step per syn_ratio synthetic steps
        use_judge = (step % (args.syn_ratio + 1) == args.syn_ratio)

        if use_judge:
            q_b, c_b, l_b, _ = next(judge_loader)
        else:
            q_b, c_b, l_b, epoch = next(syn_loader)

        q_b, c_b, l_b = q_b.to(device), c_b.to(device), l_b.to(device)

        scores = model(q_b, c_b)
        loss = F.binary_cross_entropy_with_logits(scores, l_b)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)   # tighter clip

        # Constant LR (no schedule — short fine-tune)
        optimizer.step()

        dt = time.time() - t0
        if step > 3:
            total_time += dt

        ema = 0.95
        smooth_loss = ema * smooth_loss + (1 - ema) * loss.item()
        debiased = smooth_loss / (1 - ema ** (step + 1))

        if step % 50 == 0:
            pct = min(total_time / args.time, 1.0) * 100
            src = "J" if use_judge else "S"
            print(f"\rstep {step:04d} ({pct:.1f}%) | loss: {debiased:.4f} | {src} | remaining: {max(0, args.time-total_time):.0f}s    ",
                  end="", flush=True)

        step += 1
        if step > 3 and total_time >= args.time:
            break

    print()

    # Evaluate
    print("Evaluating...")
    model.eval()
    val_ndcg = evaluate_ndcg(model, syn_data, batch_size=32, device=device)
    judge_ndcg = evaluate_judge_ndcg(model, judge_data, batch_size=32, device=device)

    print(f"\nResults vs base model (NDCG={base_ndcg:.4f}):")
    print(f"  val_ndcg (synthetic, held-out): {val_ndcg:.6f}  {'↑' if val_ndcg > base_ndcg else '↓'} vs base")
    print(f"  judge_ndcg (TRM-win train):     {judge_ndcg:.6f}  [train set — 5 unique queries]")

    torch.save({
        "model_state_dict": model.state_dict(),
        "val_ndcg": val_ndcg,
        "judge_ndcg": judge_ndcg,
        "config": cfg,
        "judge_trained": True,
        "judge_mode": "trm_wins_only",
        "finetune_lr": args.lr,
        "finetune_syn_ratio": args.syn_ratio,
    }, JUDGE_CKPT)
    print(f"Saved to {JUDGE_CKPT}")
    print(f"\nFor held-out downstream eval:")
    print(f"  uv run eval_downstream.py --resume --checkpoint {JUDGE_CKPT}")


def _pad_examples(examples: list[dict], target: int = CANDIDATE_POOL_SIZE) -> dict:
    """Inline padding (same logic as load_judge_data)."""
    qe, ce, le = [], [], []
    for ex in examples:
        q = ex["query_emb"]
        c = ex["cand_embs"]
        l = ex["labels"]
        n = c.size(0)
        if n < target:
            n_pad = target - n
            c_ext = c.repeat((n_pad // n) + 2, 1)[:n + n_pad]
            l_ext = torch.cat([l, torch.zeros(n_pad)])
        else:
            pos_idx = (l > 0.5).nonzero(as_tuple=True)[0].tolist()
            neg_idx = (l <= 0.5).nonzero(as_tuple=True)[0].tolist()
            keep = (pos_idx + neg_idx[:target - len(pos_idx)])[:target]
            c_ext = c[keep]
            l_ext = l[keep]
        qe.append(q); ce.append(c_ext); le.append(l_ext)
    return {
        "query_embeddings": torch.stack(qe),
        "candidate_embeddings": torch.stack(ce),
        "labels": torch.stack(le),
        "n_queries": len(examples),
        "pool_size": target,
        "embed_dim": examples[0]["query_emb"].size(0),
    }


if __name__ == "__main__":
    main()
