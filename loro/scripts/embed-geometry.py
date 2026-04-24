"""Quick embedding geometry comparison on a 50-doc workspace subset."""
import torch, os, time, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from prepare import load_markdown_docs, chunk_documents

workspace = os.path.join(os.path.expanduser('~'), 'workspaces', 'cog')
docs = load_markdown_docs(workspace)
chunks = chunk_documents(docs)
print(f'Loaded {len(chunks)} chunks from workspace')

subset = chunks[:50]
texts = [f"search_document: {c.get('title','')}. {c.get('text','')[:500]}" for c in subset]
print(f'Testing with {len(texts)} chunks\n')

models = [
    'nomic-ai/nomic-embed-text-v1.5',
    'Snowflake/snowflake-arctic-embed-m-v2.0',
    'BAAI/bge-m3',
]

results = []
for model_name in models:
    print(f'=== {model_name} ===')
    t0 = time.time()
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        native_dim = embs.shape[1]
        trunc_dim = min(384, native_dim)
        embs_384 = F.normalize(embs[:, :trunc_dim], p=2, dim=1)

        sim = embs_384 @ embs_384.T
        diag_mask = ~torch.eye(len(texts), dtype=torch.bool)
        off_diag = sim[diag_mask]

        elapsed = time.time() - t0
        r = {
            'model': model_name,
            'native_dim': native_dim,
            'trunc_dim': trunc_dim,
            'time': elapsed,
            'sim_mean': off_diag.mean().item(),
            'sim_std': off_diag.std().item(),
            'sim_min': off_diag.min().item(),
            'sim_max': off_diag.max().item(),
        }
        results.append(r)
        print(f'  Native dim: {native_dim}, truncated to: {trunc_dim}')
        print(f'  Time: {elapsed:.1f}s')
        print(f'  Sim: mean={r["sim_mean"]:.4f} std={r["sim_std"]:.4f} '
              f'min={r["sim_min"]:.4f} max={r["sim_max"]:.4f}')

        del model
    except Exception as e:
        print(f'  ERROR: {e}')
    print()

print('\n=== Summary ===')
print(f'{"Model":<45} {"Dim":>4} {"Mean Sim":>9} {"Std Sim":>9} {"Spread":>9}')
print('-' * 80)
for r in results:
    spread = r['sim_max'] - r['sim_min']
    print(f'{r["model"]:<45} {r["trunc_dim"]:>4} {r["sim_mean"]:>9.4f} {r["sim_std"]:>9.4f} {spread:>9.4f}')
print()
print('Higher std = more discriminative geometry = easier job for the TRM')
print('Lower mean = less "everything looks similar" = better separation')
