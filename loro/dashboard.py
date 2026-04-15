"""
Autoresearch Dashboard — live monitoring for the TRM training loop.

Single-file SPA: Python serves data + static HTML with embedded JS charts.
Auto-refreshes every 15 seconds.

Usage: uv run dashboard.py [--port 8420]
"""

import os
import json
import subprocess
import signal
import http.server
import socketserver
from urllib.parse import urlparse

PORT = 8420
DIR = os.path.dirname(os.path.abspath(__file__))


def get_results(current_phase_only=True):
    # Phase 5+: Mamba TRM results
    path = os.path.join(DIR, "results_mamba.tsv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    ndcg = float(parts[1])
                except ValueError:
                    ndcg = 0.0
                try:
                    params = int(parts[2])
                except ValueError:
                    params = 0
                rows.append({
                    "commit": parts[0],
                    "ndcg": ndcg,
                    "params": params,
                    "status": parts[3],
                    "description": parts[4] if len(parts) > 4 else "",
                    "idx": len(rows),
                })
    return rows


def get_hyperparams():
    path = os.path.join(DIR, "train_mamba.py")
    params = {}
    targets = ["D_MODEL", "D_STATE", "D_CONV", "N_LAYERS", "EXPAND_FACTOR",
               "DROPOUT", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY",
               "WARMUP_STEPS", "TRAIN_SECONDS", "CANDIDATE_POOL", "SEED"]
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                for prefix in targets:
                    if stripped.startswith(prefix) and "=" in stripped:
                        val = stripped.split("=")[1].split("#")[0].strip()
                        params[prefix] = val
    return params


def get_git_log(n=30):
    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={n}", "--format=%h|%s|%ar"],
            capture_output=True, text=True, cwd=DIR, timeout=5,
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 2)
                commits.append({
                    "hash": parts[0],
                    "message": parts[1] if len(parts) > 1 else "",
                    "ago": parts[2] if len(parts) > 2 else "",
                })
        return commits
    except Exception:
        return []


def get_ralph_status():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "claude.*program_mamba|claude.*program.md|claude.*sonnet"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        return {"running": len(pids) > 0, "pids": pids}
    except Exception:
        return {"running": False, "pids": []}


def get_judge_stats():
    path = os.path.join(DIR, "judge_data.pt")
    if not os.path.exists(path):
        return {"total": 0, "sources": {}, "error": "no file"}
    try:
        import torch
        labels = torch.load(path, map_location="cpu", weights_only=False)
        sources = {}
        for l in labels:
            w = l.get("winner", "unknown")
            sources[w] = sources.get(w, 0) + 1
        return {"total": len(labels), "sources": sources}
    except ImportError:
        # torch not available — estimate from file size
        size = os.path.getsize(path)
        return {"total": f"~{size // 1024}KB", "sources": {"(torch not available — run with: uv run dashboard.py)": 0}, "error": "no torch"}
    except Exception as e:
        return {"total": 0, "sources": {}, "error": str(e)}


def get_run_log(n=30):
    path = os.path.join(DIR, "run.log")
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""


def get_search_mode(results):
    """Infer adaptive search mode from recent results (n-autoresearch style)."""
    if len(results) < 5:
        return "explore"
    recent = results[-10:]
    keeps = sum(1 for r in recent if r["status"] == "keep")
    crashes = sum(1 for r in recent if r["status"] == "crash")
    discards = sum(1 for r in recent if r["status"] == "discard")

    if crashes >= 3:
        return "recover"
    if keeps == 0 and len(recent) >= 8:
        return "plateau"
    if keeps >= 2:
        return "exploit"
    return "explore"


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TRM Bimodal Convergence Investigation</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --muted: #8b949e; --blue: #58a6ff;
  --green: #7ee787; --red: #f85149; --orange: #f0883e; --purple: #bc8cff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; background: var(--bg); color: var(--text); padding: 16px; font-size: 13px; }
h1 { color: var(--blue); font-size: 1.3em; display: inline; }
.header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.mode-badge { padding: 3px 10px; border-radius: 12px; font-size: 0.75em; font-weight: bold; text-transform: uppercase; }
.mode-explore { background: #1f3a5f; color: var(--blue); }
.mode-exploit { background: #1a3a1a; color: var(--green); }
.mode-plateau { background: #3d1f00; color: var(--orange); }
.mode-recover { background: #3d1a1a; color: var(--red); }
.subtitle { color: var(--muted); font-size: 0.8em; }
.grid { display: grid; gap: 12px; margin-bottom: 12px; }
.grid-4 { grid-template-columns: repeat(4, 1fr); }
.grid-2 { grid-template-columns: 1fr 1fr; }
.grid-commits-results { grid-template-columns: 2fr 3fr; }
.grid-1 { grid-template-columns: 1fr; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px; min-width: 0; overflow: hidden; }
.card h2 { color: var(--blue); font-size: 0.85em; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.card.resizable { resize: both; overflow: auto; min-height: 100px; min-width: 200px; }
.metric { text-align: center; padding: 16px 12px; }
.metric-value { font-size: 2.2em; font-weight: bold; color: var(--green); line-height: 1.2; }
.metric-label { font-size: 0.75em; color: var(--muted); margin-top: 4px; }
.metric-value.running { color: var(--green); }
.metric-value.stopped { color: var(--red); }
canvas { width: 100%; display: block; }
.chart-container { width: 100%; height: 100%; min-height: 200px; }
.chart-card { resize: vertical; overflow: hidden; min-height: 250px; height: 360px; }
table { width: 100%; border-collapse: collapse; font-size: 0.85em; table-layout: auto; }
th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #21262d; }
th { color: var(--muted); font-weight: normal; position: sticky; top: 0; background: var(--surface); z-index: 1; white-space: nowrap; }
td:nth-child(1) { width: 36px; white-space: nowrap; }
td:nth-child(2) { width: 65px; white-space: nowrap; }
td:nth-child(3) { width: 55px; white-space: nowrap; }
td:nth-child(4) { width: 55px; white-space: nowrap; }
td:nth-child(5) { white-space: normal; word-wrap: break-word; line-height: 1.4; }
tr.keep { color: var(--green); } tr.discard { color: var(--muted); } tr.crash { color: var(--red); }
tr.keep td:nth-child(2) { font-weight: bold; }
.scroll { overflow-y: auto; }
.commit { padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 0.85em; }
.commit-top { display: flex; justify-content: space-between; align-items: center; }
.commit-hash { color: var(--blue); min-width: 60px; font-size: 0.9em; }
.commit-ago { color: var(--muted); font-size: 0.8em; white-space: nowrap; }
.commit-msg { color: var(--text); margin-top: 2px; white-space: normal; word-wrap: break-word; line-height: 1.4; }
.param-row { display: flex; justify-content: space-between; padding: 2px 0; }
.param-name { color: var(--muted); } .param-val { color: var(--text); font-weight: bold; }
.badge { padding: 2px 8px; border-radius: 4px; font-size: 0.7em; display: inline-block; margin: 2px 0; }
.badge-session { background: #1f3a5f; color: var(--blue); }
.badge-retro { background: #3d1f00; color: var(--orange); }
.badge-cosine { background: #3d1a1a; color: var(--red); }
.badge-trm { background: #1a3a1a; color: var(--green); }
.log-box { background: #0d1117; border: 1px solid var(--border); border-radius: 4px; padding: 8px; font-size: 0.7em; height: 100%; min-height: 80px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; color: var(--muted); font-family: inherit; }
.refresh-info { color: var(--muted); font-size: 0.7em; }
/* Resize handle styling */
::-webkit-resizer { background: var(--border); }
</style>
</head>
<body>

<div class="header">
  <h1>TRM Bimodal Convergence Investigation</h1>
  <span class="mode-badge" id="search-mode">—</span>
  <span class="subtitle">Bimodal Convergence Investigation — Identity Init, Warm-Start, Score Head Ablation</span>
  <span style="flex:1"></span>
  <span class="refresh-info" id="last-refresh">—</span>
</div>

<!-- Metrics row -->
<div class="grid grid-4">
  <div class="card metric">
    <div class="metric-value" id="best-ndcg">—</div>
    <div class="metric-label">Best NDCG@10</div>
  </div>
  <div class="card metric">
    <div class="metric-value" id="total-exp">—</div>
    <div class="metric-label">Experiments</div>
  </div>
  <div class="card metric">
    <div class="metric-value" id="ralph-status">—</div>
    <div class="metric-label">Ralph</div>
  </div>
  <div class="card metric">
    <div class="metric-value" id="judge-count" style="color:var(--blue)">—</div>
    <div class="metric-label">Judge Labels</div>
  </div>
</div>

<!-- Chart -->
<div class="grid grid-1">
  <div class="card chart-card">
    <h2>NDCG@10 Convergence</h2>
    <div class="chart-container"><canvas id="chart"></canvas></div>
  </div>
</div>

<!-- Middle row: params + judge + log -->
<div class="grid" style="grid-template-columns: 1fr 1fr 1.5fr;">
  <div class="card resizable">
    <h2>Architecture</h2>
    <div id="params"></div>
  </div>
  <div class="card resizable">
    <h2>Training Data</h2>
    <div id="judge-sources"></div>
  </div>
  <div class="card resizable">
    <h2>Live Progress</h2>
    <div class="log-box" id="run-log">—</div>
  </div>
</div>

<!-- Bottom: commits + results (full width, stacked) -->
<div class="grid grid-2">
  <div class="card resizable" style="height:400px;">
    <h2>Ralph's Reasoning</h2>
    <div class="scroll" id="commits" style="height:calc(100% - 36px);"></div>
  </div>
  <div class="card resizable" style="height:400px;">
    <h2>Results (newest first)</h2>
    <div class="scroll" id="results" style="height:calc(100% - 36px);"></div>
  </div>
</div>

<script>
async function fetchData() {
  const res = await fetch('/api/data');
  return await res.json();
}

function drawChart(canvas, results) {
  const dpr = window.devicePixelRatio || 1;
  const container = canvas.parentElement;
  const W = container.clientWidth;
  const H = container.clientHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const valid = results.filter(r => r.ndcg > 0.1);
  if (!valid.length) return;

  const ndcgs = valid.map(r => r.ndcg);
  // Zoom into the interesting range: use 10th percentile as floor (avoids early outliers stretching the axis)
  const sorted = [...ndcgs].sort((a, b) => a - b);
  const p10 = sorted[Math.floor(sorted.length * 0.1)] || sorted[0];
  const minY = Math.max(p10 - 0.02, 0);
  const maxY = Math.max(...ndcgs) + 0.01;
  const pad = { l: 56, r: 16, t: 16, b: 24 };
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const toX = i => pad.l + (i / Math.max(valid.length - 1, 1)) * cw;
  const toY = v => pad.t + ch * (1 - (v - minY) / (maxY - minY));

  // Grid lines + labels
  ctx.textBaseline = 'middle'; ctx.font = '10px monospace';
  for (let i = 0; i <= 5; i++) {
    const y = pad.t + ch * i / 5;
    ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#8b949e';
    ctx.fillText((maxY - (maxY - minY) * i / 5).toFixed(3), 2, y);
  }

  // Cosine baseline
  const cosine = results.find(r => r.commit === 'COSINE');
  if (cosine) {
    const y = toY(cosine.ndcg);
    ctx.strokeStyle = '#f8514944'; ctx.setLineDash([6, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = '#f85149'; ctx.font = '9px monospace';
    ctx.fillText('cosine ' + cosine.ndcg.toFixed(3), pad.l + 4, y - 8);
  }

  // Scatter points
  valid.forEach((r, i) => {
    const x = toX(i), y = toY(r.ndcg);
    ctx.beginPath();
    ctx.arc(x, y, r.status === 'keep' ? 4 : 2, 0, Math.PI * 2);
    ctx.fillStyle = r.status === 'keep' ? '#7ee787' : r.status === 'crash' ? '#f85149' : '#484f58';
    ctx.fill();
  });

  // Best-so-far line
  let best = 0;
  const pts = [];
  valid.forEach((r, i) => {
    if (r.status === 'keep' && r.ndcg > best) {
      best = r.ndcg;
      pts.push({ x: toX(i), y: toY(best) });
    }
  });
  if (pts.length > 1) {
    ctx.strokeStyle = '#7ee78788'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
    pts.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.stroke();
  }

  // Current best label
  if (best > 0) {
    ctx.fillStyle = '#7ee787'; ctx.font = 'bold 11px monospace';
    ctx.fillText(best.toFixed(4), W - pad.r - 60, toY(best) - 8);
  }
}

function render(data) {
  const keeps = data.results.filter(r => r.status === 'keep');
  const best = keeps.length ? Math.max(...keeps.map(r => r.ndcg)) : 0;
  document.getElementById('best-ndcg').textContent = best.toFixed(4);
  document.getElementById('total-exp').textContent = data.results.length + ' / ' + data.total_experiments;

  const rs = document.getElementById('ralph-status');
  rs.textContent = data.ralph.running ? 'RUNNING' : 'STOPPED';
  rs.className = 'metric-value ' + (data.ralph.running ? 'running' : 'stopped');
  document.getElementById('judge-count').textContent = data.judge.total;

  // Search mode
  const mode = data.search_mode;
  const mb = document.getElementById('search-mode');
  mb.textContent = mode;
  mb.className = 'mode-badge mode-' + mode;

  // Chart
  drawChart(document.getElementById('chart'), data.results);

  // Params
  document.getElementById('params').innerHTML = Object.entries(data.params)
    .map(([k, v]) => `<div class="param-row"><span class="param-name">${k}</span><span class="param-val">${v}</span></div>`)
    .join('');

  // Judge sources
  const badgeClass = {'SESSION_MINED': 'badge-session', 'RETROSPECTIVE': 'badge-retro', 'cosine': 'badge-cosine', 'trm': 'badge-trm'};
  const judgeDiv = document.getElementById('judge-sources');
  if (data.judge.error) {
    judgeDiv.innerHTML = `<div style="color:var(--red)">Error: ${data.judge.error}</div><div style="color:var(--muted);font-size:0.8em;margin-top:4px">Launch with: <code>uv run dashboard.py</code></div>`;
  } else {
    judgeDiv.innerHTML =
      `<div style="margin-bottom:6px;color:var(--muted)">Total: <strong style="color:var(--text)">${data.judge.total}</strong> labels</div>` +
      Object.entries(data.judge.sources)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `<div><span class="badge ${badgeClass[k] || 'badge-session'}">${k}</span> <strong>${v}</strong></div>`)
        .join('');
  }

  // Log
  const logBox = document.getElementById('run-log');
  if (data.run_log) {
    logBox.textContent = data.run_log;
    logBox.scrollTop = logBox.scrollHeight;
  }

  // Commits
  document.getElementById('commits').innerHTML = data.commits
    .map(c => `<div class="commit"><div class="commit-top"><span class="commit-hash">${c.hash}</span><span class="commit-ago">${c.ago}</span></div><div class="commit-msg">${c.message}</div></div>`)
    .join('');

  // Results table
  const sorted = [...data.results].reverse();
  document.getElementById('results').innerHTML =
    '<table><tr><th>#</th><th>NDCG</th><th>Params</th><th>Status</th><th>Description</th></tr>' +
    sorted.map(r =>
      `<tr class="${r.status}"><td>${r.idx}</td><td>${r.ndcg.toFixed(4)}</td><td>${(r.params/1e6).toFixed(2)}M</td><td>${r.status}</td><td>${r.description}</td></tr>`
    ).join('') + '</table>';

  document.getElementById('last-refresh').textContent = 'Updated ' + new Date().toLocaleTimeString();
}

async function refresh() {
  try { render(await fetchData()); } catch(e) { console.error(e); }
}
refresh();
setInterval(refresh, 15000);

// Redraw chart when its container is resized
let lastData = null;
const chartCard = document.querySelector('.chart-card');
if (chartCard) {
  new ResizeObserver(() => {
    if (lastData) drawChart(document.getElementById('chart'), lastData.results);
  }).observe(chartCard);
}
// Store data for resize redraws
const origRender = render;
render = function(data) { lastData = data; origRender(data); };
</script>
</body>
</html>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/data":
            results = get_results()
            data = {
                "results": results,
                "total_experiments": len(results),
                "params": get_hyperparams(),
                "commits": get_git_log(30),
                "ralph": get_ralph_status(),
                "judge": get_judge_stats(),
                "run_log": get_run_log(20),
                "search_mode": get_search_mode(results),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            return

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoresearch Dashboard")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print(f"Dashboard: http://localhost:{args.port}")
        print(f"Auto-refreshes every 15s. Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
