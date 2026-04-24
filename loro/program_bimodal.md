# Ralph Program v3 — Bimodal Convergence Investigation

You are Ralph, the autoresearch agent. Your mission: **investigate and resolve the bimodal convergence pattern** discovered in the TRM training pipeline.

## Background

A systematic ablation study (2026-04-11/12) found that the 2.28M-parameter Mamba TRM consistently converges to one of two basins:

- **Basin A (~52% of seeds):** NDCG@10 = 0.72 ± 0.03
- **Basin B (~35% of seeds):** NDCG@10 = 0.86 ± 0.02
- **Transition (~13%):** 0.78-0.83

This bimodality is **invariant** to:
- Embedding model (nomic-embed-text vs bge-m3)
- Loss function (cross-entropy vs InfoNCE)
- SSM state dimension (D_STATE 1 through 16)
- Training data size (950 pairs vs 7K+ merged pairs)

See: `.cog/mem/semantic/insights/bimodal-convergence-ablation-study.cog.md`

## Your Experiments

Run these in priority order. Each experiment should:
1. State the hypothesis clearly
2. Change ONE variable
3. Run 5-10 seeds
4. Report mean, σ, and basin distribution (Lo/Hi split)
5. Write results to `results_bimodal.tsv`
6. Update the dashboard at `dashboard_bimodal.html`

### Experiment 1: Identity Initialization of in_proj

**Hypothesis:** The bimodality is caused by random initialization of `in_proj` (Linear 384→768). If we initialize it as near-identity (embedding passthrough), all seeds should converge to Basin B because the starting point is already meaningful.

**Method:**
```python
# In MambaTRM.__init__, after self.in_proj is created:
with torch.no_grad():
    # Initialize as [I; 0] so first half passes embeddings through
    nn.init.eye_(self.in_proj.weight[:384, :])
    nn.init.zeros_(self.in_proj.weight[384:, :])
```

Run 10 seeds (1-10) with this init. Compare to standard Xavier init.

### Experiment 2: Warm-Start from Basin B

**Hypothesis:** If Basin B represents a stable attractor, warm-starting from a Basin B checkpoint with different seeds should keep all runs in Basin B.

**Method:**
1. Load `best_model_mamba.pt` (a Basin B model from seed 42)
2. For each of 10 seeds: perturb the weights slightly (add noise σ=0.01), then retrain for 3200 steps
3. Report whether all retrained models stay in Basin B

### Experiment 3: Score Head Ablation

**Hypothesis:** The bimodality might be in the learned score head, not the SSM. If we replace the learned score head with fixed cosine similarity, does bimodality disappear?

**Method:**
```python
# Replace score head with cosine similarity
# Instead of: scores = self.score_head(h)
# Use: scores = F.cosine_similarity(h.unsqueeze(2), candidates, dim=-1)
```

Run 10 seeds. If bimodality disappears, the score head initialization is the mechanism.

### Experiment 4: Projection Dimension Sweep

**Hypothesis:** The in_proj maps 384→768 (2x expansion). What if we try 384→384 (no expansion) or 384→1536 (4x)? The expansion ratio determines how much room the model has for the initial projection to vary.

**Method:** Sweep EXPAND_FACTOR in {1, 2, 4} with 5 seeds each. (Note: current EXPAND_FACTOR=1 means d_inner=384. The in_proj is 384→384*2=768 because it splits into SSM input and gate.)

### Experiment 5: Training Data Curriculum

**Hypothesis:** Shuffled training might prevent the SSM from learning temporal patterns. Training on sessions in chronological order might give the SSM better temporal signal and reduce basin sensitivity.

**Method:** Modify the dataloader to present sessions chronologically rather than shuffled. Run 10 seeds.

### Experiment 6: Multi-Seed Ensemble

**Hypothesis:** Instead of picking one seed, average predictions from the top-3 seeds. This should give Basin B performance with much higher reliability.

**Method:**
1. Train 10 seeds
2. For each pair of 3 models, compute ensemble NDCG (average scores before ranking)
3. Report the ensemble's NDCG vs best individual seed

## Dashboard

Create `dashboard_bimodal.html` — a self-contained HTML file with:

1. **Basin Distribution Chart** — histogram of NDCG scores across all experiments, colored by experiment
2. **Ablation Summary Table** — experiment name, N seeds, mean, σ, Lo%, Hi%, verdict
3. **Convergence Curves** — if you log per-step NDCG, show the training curves colored by final basin
4. **Live Results** — auto-refresh from `results_bimodal.tsv`

Use Chart.js or inline SVG. The file should be viewable by opening it in a browser.

## Results Format

Append to `results_bimodal.tsv`:
```
experiment	seed	ndcg	cosine_baseline	d_state	expand	init_type	notes
identity_init	1	0.XXXX	0.XXXX	4	1	identity	Experiment 1
identity_init	2	0.XXXX	0.XXXX	4	1	identity	Experiment 1
warmstart_b	1	0.XXXX	0.XXXX	4	1	warmstart	Experiment 2
...
```

## Constraints

- **DO NOT modify prepare.py** — use the existing data.pt
- **DO modify train_mamba.py** — but restore it to the default config after each experiment
- **Time budget:** Each experiment should complete in <20 minutes (10 seeds × ~90s)
- **Seed range:** Use seeds 1-10 for consistency across experiments
- **Always report honestly** — negative results are valuable

## Success Criteria

The investigation succeeds if we either:
1. **Find the mechanism** — identify which component's initialization causes the bimodal split
2. **Find a reliable fix** — a configuration that achieves σ < 0.02 across 10+ seeds
3. **Characterize the phenomenon** — enough data to write the "Bimodal Convergence in Small SSMs" section of the paper with confidence

Any of these is a win.
