#!/usr/bin/env python3
"""
TRM Weight Export — one-time script to export MambaTRM weights to Go-readable binary.

Exports:
  1. trm_weights.bin    — all model weights in TRM1 binary format
  2. trm_embeddings.bin — embedding index in EMB1 binary format
  3. trm_chunks.json    — copy of chunk metadata
  4. trm_reference.json — reference outputs for Go validation

Binary formats:

  trm_weights.bin:
    4 bytes: magic "TRM1"
    4 bytes: uint32 number of tensors
    For each tensor:
      4 bytes: uint32 name length
      N bytes: name string (UTF-8)
      4 bytes: uint32 number of dimensions
      4*ndim bytes: uint32 shape for each dimension
      4*numel bytes: float32 data (row-major, little-endian)

  trm_embeddings.bin:
    4 bytes: magic "EMB1"
    4 bytes: uint32 num_chunks
    4 bytes: uint32 dim (384)
    num_chunks * dim * 4 bytes: float32 data (row-major, little-endian)

Usage:
    cd apps/cogos-v3/autoresearch
    uv run trm_export.py
    uv run trm_export.py --output-dir /path/to/output
"""

import os
import sys
import json
import struct
import shutil
import argparse
import numpy as np

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model_mamba.pt")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cogos-autoresearch")
EMBED_PATH = os.path.join(CACHE_DIR, "embeddings.pt")
CHUNKS_PATH = os.path.join(CACHE_DIR, "embed_index", "chunks.json")

EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Weight export (TRM1 format)
# ---------------------------------------------------------------------------

def write_trm_weights(state_dict: dict, output_path: str):
    """Write all model weights in TRM1 binary format."""
    tensors = []
    for name, param in state_dict.items():
        arr = param.cpu().float().numpy()
        tensors.append((name, arr))

    with open(output_path, "wb") as f:
        # Magic
        f.write(b"TRM1")
        # Number of tensors
        f.write(struct.pack("<I", len(tensors)))

        for name, arr in tensors:
            name_bytes = name.encode("utf-8")
            # Name length + name
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            # Number of dimensions
            f.write(struct.pack("<I", len(arr.shape)))
            # Shape
            for dim in arr.shape:
                f.write(struct.pack("<I", dim))
            # Data (row-major, little-endian float32)
            f.write(arr.astype("<f4").tobytes())

    total_params = sum(arr.size for _, arr in tensors)
    print(f"  Wrote {len(tensors)} tensors, {total_params:,} parameters")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Print tensor summary
    for name, arr in tensors:
        print(f"    {name:45s} {str(list(arr.shape)):20s} {arr.size:>10,}")


# ---------------------------------------------------------------------------
# Embedding export (EMB1 format)
# ---------------------------------------------------------------------------

def write_embeddings(embeddings: torch.Tensor, output_path: str):
    """Write embeddings in EMB1 binary format."""
    n_chunks, dim = embeddings.shape
    arr = embeddings.cpu().float().numpy()

    with open(output_path, "wb") as f:
        f.write(b"EMB1")
        f.write(struct.pack("<I", n_chunks))
        f.write(struct.pack("<I", dim))
        f.write(arr.astype("<f4").tobytes())

    print(f"  {n_chunks} chunks x {dim} dims")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


# ---------------------------------------------------------------------------
# Reference output generation
# ---------------------------------------------------------------------------

def generate_reference(state_dict: dict, config: dict, embeddings: torch.Tensor,
                       output_path: str):
    """Generate reference outputs for Go validation."""
    # Import the model class
    sys.path.insert(0, SCRIPT_DIR)
    from train_mamba import MambaTRM

    model = MambaTRM(
        d_model=config.get("d_model", 384),
        d_state=config.get("d_state", 4),
        d_conv=config.get("d_conv", 2),
        n_layers=config.get("n_layers", 2),
        expand=config.get("expand", 1),
        dropout=0.0,  # no dropout in inference
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Generate a deterministic test query embedding
    # Use the first embedding as a "test query" for reproducibility
    test_query = embeddings[0].clone()

    # Take first 64 candidate embeddings
    n_cands = min(64, embeddings.shape[0])
    candidates = embeddings[:n_cands].clone()

    reference = {
        "config": config,
        "test_query": test_query.tolist(),
        "candidate_indices": list(range(n_cands)),
        "n_candidates": n_cands,
    }

    with torch.no_grad():
        # Test 1: Score candidates without light cone (fresh state)
        # Use step() to process the query, then score
        context_no_lc, states_no_lc = model.step(
            test_query.unsqueeze(0),  # (1, 384)
            torch.tensor([0]),        # event_type=0 (query)
            states=None,
        )

        scores_no_lc = model.score_candidates(
            context_no_lc,
            candidates.unsqueeze(0),  # (1, N, 384)
        )

        reference["scores_no_lightcone"] = scores_no_lc[0].tolist()
        reference["context_no_lightcone"] = context_no_lc[0].tolist()

        # Save the SSM states for reference
        reference["states_no_lightcone"] = []
        for s in states_no_lc:
            reference["states_no_lightcone"].append(s[0].tolist())  # (d_inner, d_state)

        # Test 2: Feed a second event (simulating light cone with prior state)
        second_event = embeddings[1].clone()
        context_with_lc, states_with_lc = model.step(
            second_event.unsqueeze(0),
            torch.tensor([1]),  # event_type=1 (retrieval)
            states=states_no_lc,
        )

        scores_with_lc = model.score_candidates(
            context_with_lc,
            candidates.unsqueeze(0),
        )

        reference["second_event"] = second_event.tolist()
        reference["scores_with_lightcone"] = scores_with_lc[0].tolist()
        reference["context_with_lightcone"] = context_with_lc[0].tolist()

        reference["states_with_lightcone"] = []
        for s in states_with_lc:
            reference["states_with_lightcone"].append(s[0].tolist())

    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"  Reference scores (no LC): min={min(reference['scores_no_lightcone']):.4f}, "
          f"max={max(reference['scores_no_lightcone']):.4f}")
    print(f"  Reference scores (with LC): min={min(reference['scores_with_lightcone']):.4f}, "
          f"max={max(reference['scores_with_lightcone']):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export MambaTRM weights for Go")
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR,
                        help="Output directory for exported files")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to best_model_mamba.pt")
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip reference output generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load and export model weights
    print(f"\n=== Loading model from {args.model} ===")
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"]
    config = ckpt.get("config", {})

    print(f"Config: {config}")
    print(f"NDCG: {ckpt.get('ndcg', 'N/A')}")

    weights_path = os.path.join(args.output_dir, "trm_weights.bin")
    print(f"\n=== Exporting weights to {weights_path} ===")
    write_trm_weights(state_dict, weights_path)

    # 2. Load and export embeddings
    print(f"\n=== Loading embeddings from {EMBED_PATH} ===")
    if not os.path.exists(EMBED_PATH):
        print(f"ERROR: Embeddings not found at {EMBED_PATH}")
        print("Run: cd autoresearch && uv run embed_index.py")
        sys.exit(1)

    embeddings = torch.load(EMBED_PATH, map_location="cpu", weights_only=True)
    print(f"Shape: {embeddings.shape}")

    emb_path = os.path.join(args.output_dir, "trm_embeddings.bin")
    print(f"\n=== Exporting embeddings to {emb_path} ===")
    write_embeddings(embeddings, emb_path)

    # 3. Copy chunks.json
    print(f"\n=== Copying chunks metadata ===")
    if not os.path.exists(CHUNKS_PATH):
        print(f"ERROR: chunks.json not found at {CHUNKS_PATH}")
        sys.exit(1)

    chunks_dest = os.path.join(args.output_dir, "trm_chunks.json")
    shutil.copy2(CHUNKS_PATH, chunks_dest)
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks copied")

    # 4. Generate reference outputs
    if not args.skip_reference:
        ref_path = os.path.join(args.output_dir, "trm_reference.json")
        print(f"\n=== Generating reference outputs to {ref_path} ===")
        generate_reference(state_dict, config, embeddings, ref_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"Export complete! Files in {args.output_dir}:")
    for fname in ["trm_weights.bin", "trm_embeddings.bin", "trm_chunks.json", "trm_reference.json"]:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                print(f"  {fname:30s} {size / 1024 / 1024:.2f} MB")
            else:
                print(f"  {fname:30s} {size / 1024:.1f} KB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
