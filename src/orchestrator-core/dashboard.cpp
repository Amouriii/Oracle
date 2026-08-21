// The Oracle dashboard: one self-contained page served at "/".
//
// It is deliberately dependency-free -- no CDN, no build step -- because the
// master often runs on an isolated Thunderbolt subnet with no route to the
// internet.  It polls /cluster and /health and renders the mesh, the model, the
// request queue, per-node CPU/RAM/GPU, the network links and the security state.
#include "oracle/orch/pipeline_orchestrator.hpp"

namespace oracle {

const char* dashboard_html() {
  return R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oracle</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #131822; --panel2: #1a2130; --line: #232c3d;
    --text: #e6edf7; --dim: #8b98ad; --accent: #4ea1ff; --ok: #3ddc97;
    --warn: #ffb454; --bad: #ff6b6b; --radius: 10px;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 14px 22px; border-bottom: 1px solid var(--line); background: var(--panel); }
  h1 { font-size: 18px; margin: 0; letter-spacing: .5px; }
  h1 span { color: var(--accent); }
  .pill { padding: 3px 10px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line);
    background: var(--panel2); color: var(--dim); }
  .pill.ok { color: var(--ok); border-color: #22503c; }
  .pill.warn { color: var(--warn); border-color: #5a4420; }
  .pill.bad { color: var(--bad); border-color: #5c2626; }
  main { padding: 18px 22px 40px; max-width: 1400px; margin: 0 auto; }
  .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 14px 16px; }
  .card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--dim);
    margin: 0 0 10px; font-weight: 600; }
  .stat { font-size: 26px; font-weight: 600; }
  .sub { color: var(--dim); font-size: 12px; }
  section { margin-top: 22px; }
  section > h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 1px;
    color: var(--dim); margin: 0 0 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line);
    white-space: nowrap; }
  th { color: var(--dim); font-weight: 600; font-size: 11px; text-transform: uppercase;
    letter-spacing: .6px; }
  .scroll { overflow-x: auto; background: var(--panel); border: 1px solid var(--line);
    border-radius: var(--radius); }
  .bar { height: 6px; background: var(--panel2); border-radius: 4px; overflow: hidden;
    min-width: 90px; }
  .bar > i { display: block; height: 100%; background: var(--accent); }
  .bar > i.warn { background: var(--warn); }
  .bar > i.bad { background: var(--bad); }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .dot.ok { background: var(--ok); } .dot.warn { background: var(--warn); }
  .dot.bad { background: var(--bad); } .dot.dim { background: var(--dim); }
  .pipeline { display: flex; gap: 10px; flex-wrap: wrap; align-items: stretch; }
  .stage { flex: 1 1 180px; background: var(--panel); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 12px 14px; position: relative; }
  .stage .idx { color: var(--dim); font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
  .stage .layers { font-size: 18px; font-weight: 600; margin: 4px 0; }
  .tag { display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 4px;
    background: var(--panel2); color: var(--dim); border: 1px solid var(--line); margin-right: 4px; }
  .log { font: 12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; max-height: 240px;
    overflow-y: auto; }
  .log div { border-bottom: 1px solid var(--line); padding: 4px 0; }
  .log .alert { color: var(--bad); } .log .warning { color: var(--warn); }
  .log .info { color: var(--dim); }
  footer { color: var(--dim); font-size: 12px; padding: 0 22px 24px; text-align: center; }
  .err { background: #2a1618; border: 1px solid #5c2626; color: #ffb4b4; padding: 10px 14px;
    border-radius: var(--radius); margin-bottom: 14px; }
  input { background: var(--panel2); border: 1px solid var(--line); color: var(--text);
    border-radius: 6px; padding: 5px 9px; font: inherit; font-size: 12px; width: 240px; }
</style>
</head>
<body>
<header>
  <h1>ORACLE<span>.</span></h1>
  <span class="pill" id="status">connecting</span>
  <span class="pill" id="model">model &mdash;</span>
  <span class="pill" id="uptime">uptime &mdash;</span>
  <span style="flex:1"></span>
  <input id="key" type="password" placeholder="API key (optional)" autocomplete="off">
  <span class="pill" id="tick">&mdash;</span>
</header>
<main>
  <div id="error"></div>

  <div class="grid">
    <div class="card"><h2>Workers</h2><div class="stat" id="k-workers">&mdash;</div>
      <div class="sub" id="k-workers-sub">&nbsp;</div></div>
    <div class="card"><h2>Active requests</h2><div class="stat" id="k-active">&mdash;</div>
      <div class="sub" id="k-active-sub">&nbsp;</div></div>
    <div class="card"><h2>Queued</h2><div class="stat" id="k-queued">&mdash;</div>
      <div class="sub" id="k-queued-sub">&nbsp;</div></div>
    <div class="card"><h2>Throughput</h2><div class="stat" id="k-tps">&mdash;</div>
      <div class="sub" id="k-tps-sub">&nbsp;</div></div>
    <div class="card"><h2>Network</h2><div class="stat" id="k-net">&mdash;</div>
      <div class="sub" id="k-net-sub">&nbsp;</div></div>
    <div class="card"><h2>Security</h2><div class="stat" id="k-sec">&mdash;</div>
      <div class="sub" id="k-sec-sub">&nbsp;</div></div>
  </div>

  <section><h2>Pipeline</h2><div class="pipeline" id="pipeline"></div></section>

  <section><h2>Model</h2><div class="scroll"><table id="model-table"></table></div></section>

  <section><h2>Nodes</h2><div class="scroll"><table>
    <thead><tr><th>Node</th><th>Host</th><th>Role</th><th>Layers</th><th>State</th>
      <th>CPU</th><th>RAM</th><th>GPU</th><th>Active</th><th>Link</th><th>Heartbeat</th></tr></thead>
    <tbody id="nodes"></tbody></table></div></section>

  <section><h2>Requests</h2><div class="scroll"><table>
    <thead><tr><th>ID</th><th>State</th><th>Key</th><th>Priority</th><th>Prompt</th>
      <th>Generated</th><th>Queued</th><th>Run</th><th>Note</th></tr></thead>
    <tbody id="requests"></tbody></table></div></section>

  <section><h2>Security events</h2><div class="card log" id="events">
    <div class="info">admin API key required to see the audit trail</div></div></section>
</main>
<footer>Oracle distributed inference &mdash; polling /cluster every 2s</footer>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const keyInput = $("key");
keyInput.value = localStorage.getItem("oracle-key") || "";
keyInput.addEventListener("change", () => {
  localStorage.setItem("oracle-key", keyInput.value.trim());
  refresh();
});

function bytes(n) {
  if (n === undefined || n === null) return "—";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let v = Number(n), i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return v.toFixed(v < 10 && i > 0 ? 1 : 0) + " " + u[i];
}
function pct(a, b) { return (!b || b <= 0) ? 0 : Math.max(0, Math.min(100, (a / b) * 100)); }
function bar(fraction) {
  const p = Math.max(0, Math.min(100, fraction));
  const cls = p > 90 ? "bad" : p > 70 ? "warn" : "";
  return `<div class="bar"><i class="${cls}" style="width:${p}%"></i></div>`;
}
function dur(s) {
  s = Number(s) || 0;
  const d = Math.floor(s / 86400), h = Math.floor(s / 3600) % 24;
  const m = Math.floor(s / 60) % 60, sec = Math.floor(s) % 60;
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function esc(s) {
  return String(s === undefined || s === null ? "" : s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function stateClass(state) {
  if (state === "ready") return "ok";
  if (state === "busy") return "ok";
  if (state === "degraded" || state === "joining") return "warn";
  if (state === "dead") return "bad";
  return "dim";
}

async function get(path) {
  const headers = {};
  const k = keyInput.value.trim();
  if (k) headers["Authorization"] = "Bearer " + k;
  const r = await fetch(path, { headers });
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { const j = await r.json(); if (j.error && j.error.message) msg = j.error.message; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
}

function renderPipeline(d) {
  const byId = {};
  (d.workers || []).forEach((w) => { byId[w.id] = w; });
  $("pipeline").innerHTML = (d.pipeline || []).map((s, i) => {
    const w = byId[s.node] || {};
    const tags = [];
    if (s.embed) tags.push("embed");
    if (s.lm_head) tags.push("lm_head");
    return `<div class="stage">
      <div class="idx">stage ${i} &middot; node ${s.node}</div>
      <div class="layers">layers ${s.layers.start}–${s.layers.end - 1}</div>
      <div class="sub"><span class="dot ${stateClass(w.state)}"></span>${esc(w.state || "unknown")}
        &middot; ${esc(w.host || "")}</div>
      <div style="margin-top:8px">${tags.map((t) => `<span class="tag">${t}</span>`).join("")}
        <span class="tag">${s.layers.count} layers</span></div>
    </div>`;
  }).join("") || '<div class="sub">no pipeline</div>';
}

function renderModel(d) {
  const m = d.model || {};
  const rows = [
    ["Name", esc(m.name)], ["Architecture", esc(m.architecture || "—")],
    ["Quantisation", esc(m.quantization || "—")],
    ["Parameters", m.parameters ? Number(m.parameters).toLocaleString() : "—"],
    ["Bits / weight", m.bits_per_weight ? Number(m.bits_per_weight).toFixed(2) : "—"],
    ["Layers", m.n_layers], ["Hidden dim", m.hidden_dim], ["Vocab", m.n_vocab],
    ["Context", m.context_length], ["Weights", bytes(m.weight_bytes)],
    ["Per layer", bytes(m.bytes_per_layer)],
    ["Runner", esc(m.runner) + " / " + esc(m.compute_backend) + " ×" + esc(m.compute_threads)],
    ["Path", esc(m.path || "(none)")],
  ];
  $("model-table").innerHTML = "<tbody>" + rows.map(
    (r) => `<tr><th style="width:180px">${r[0]}</th><td>${r[1]}</td></tr>`).join("") + "</tbody>";
}

function renderNodes(d) {
  const links = {};
  ((d.network || {}).links || []).forEach((l) => { links[l.node] = l; });
  $("nodes").innerHTML = (d.workers || []).map((w) => {
    const ramUsed = pct(w.ram.total - w.ram.free, w.ram.total);
    const cpu = pct((w.cpu.load || 0) * 100, 100);
    const link = links[w.id];
    const gpu = w.gpu && w.gpu.present
      ? `${bar(pct(w.gpu.total - w.gpu.free, w.gpu.total))}<span class="sub">${bytes(w.gpu.total)}</span>`
      : '<span class="sub">none</span>';
    const hb = w.heartbeat.age_ms < 0 ? "never" : Math.round(w.heartbeat.age_ms) + " ms";
    return `<tr>
      <td>${w.id}</td><td>${esc(w.host)}</td><td>${esc(w.role)}</td>
      <td>${w.layers.start}–${w.layers.end - 1} <span class="sub">(${w.layers.count})</span></td>
      <td><span class="dot ${w.link_up === false ? "bad" : stateClass(w.state)}"></span>${esc(w.state)}${
        w.link_up === false ? ' <span class="sub">link down</span>' : ""}</td>
      <td>${bar(cpu)}<span class="sub">${(w.cpu.load * 100).toFixed(0)}% of ${w.cpu.cores}c</span></td>
      <td>${bar(ramUsed)}<span class="sub">${bytes(w.ram.free)} free</span></td>
      <td>${gpu}</td>
      <td>${w.load.active}/${w.load.max_concurrent}
        <span class="sub">q${w.load.queued}</span></td>
      <td>${link ? (link.last_send_ms.toFixed(2) + " ms") : "—"}
        <span class="sub">${link ? bytes(link.bytes_sent) : ""}</span></td>
      <td>${hb} <span class="sub">${w.heartbeat.missed ? "miss " + w.heartbeat.missed : ""}</span></td>
    </tr>`;
  }).join("") || '<tr><td colspan="11" class="sub">no nodes</td></tr>';
}

function renderRequests(d) {
  const s = d.scheduler || {};
  const rows = (s.active || []).concat((s.recent || []).slice().reverse());
  const seen = new Set();
  $("requests").innerHTML = rows.filter((r) => {
    if (seen.has(r.id)) return false;
    seen.add(r.id); return true;
  }).slice(0, 20).map((r) => `<tr>
      <td>${esc(r.id)}</td>
      <td><span class="dot ${r.state === "completed" ? "ok" : r.state === "running" ? "ok"
        : r.state === "queued" ? "warn" : "bad"}"></span>${esc(r.state)}</td>
      <td>${esc(r.api_key)}</td><td>${r.priority}</td><td>${r.prompt_tokens}</td>
      <td>${r.generated_tokens}</td><td>${r.queued_ms.toFixed(0)} ms</td>
      <td>${r.run_ms.toFixed(0)} ms</td><td class="sub">${esc(r.error || "")}</td>
    </tr>`).join("") || '<tr><td colspan="9" class="sub">no requests yet</td></tr>';
}

function renderSecurity(d) {
  const sec = d.security || {};
  const traffic = sec.traffic || {};
  $("k-sec").textContent = sec.auth_required ? "enforced" : "open";
  const bits = [];
  if (sec.api_keys !== undefined) bits.push(sec.api_keys + " keys");
  if (sec.worker_auth !== undefined) bits.push("worker auth " + (sec.worker_auth ? "on" : "off"));
  if (traffic.rejected !== undefined) bits.push(traffic.rejected + " rejected");
  else if (sec.rejected !== undefined) bits.push(sec.rejected + " rejected");
  $("k-sec-sub").textContent = bits.join(" · ") || " ";
  if (sec.recent) {
    $("events").innerHTML = sec.recent.slice().reverse().map((e) =>
      `<div class="${esc(e.severity)}">${esc(e.at)} [${esc(e.category)}] ${esc(e.subject)} &mdash; ${esc(e.detail)}</div>`
    ).join("") || '<div class="info">no events</div>';
  }
}

function render(d) {
  const c = d.cluster || {};
  const workers = d.workers || [];
  // Mirrors the server's own definition of healthy: a live process *and* an
  // open activation link.
  const alive = workers.filter(
    (w) => w.link_up !== false && (w.state === "ready" || w.state === "busy")).length;
  const stats = (d.scheduler || {}).stats || {};
  const net = d.network || {};

  const st = $("status");
  st.textContent = alive === workers.length ? "healthy" : alive === 0 ? "down" : "degraded";
  st.className = "pill " + (alive === workers.length ? "ok" : alive === 0 ? "bad" : "warn");
  $("model").textContent = "model " + ((d.model || {}).name || "—");
  $("uptime").textContent = "uptime " + dur(c.uptime_seconds);

  $("k-workers").textContent = alive + " / " + workers.length;
  $("k-workers-sub").textContent = c.name + " · node " + c.self + " (" + c.role + ")";
  $("k-active").textContent = stats.running !== undefined ? stats.running : "—";
  $("k-active-sub").textContent = (stats.completed || 0) + " completed · " +
    (stats.failed || 0) + " failed";
  $("k-queued").textContent = stats.queued !== undefined ? stats.queued : "—";
  $("k-queued-sub").textContent = "avg wait " + Math.round(stats.avg_queue_ms || 0) + " ms";
  $("k-tps").textContent = (stats.tokens_per_second || 0).toFixed(1);
  $("k-tps-sub").textContent = (stats.generated_tokens || 0) + " tokens generated";
  $("k-net").textContent = bytes(net.bytes_sent);
  $("k-net-sub").textContent = (net.connected_peers || 0) + "/" + (net.peers || 0) +
    " links · " + (net.frames_sent || 0) + " frames · " + (net.errors || 0) + " errors";

  renderPipeline(d);
  renderModel(d);
  renderNodes(d);
  renderRequests(d);
  renderSecurity(d);
}

let timer = null;
async function refresh() {
  try {
    const d = await get("/cluster");
    $("error").innerHTML = "";
    render(d);
    $("tick").textContent = new Date().toLocaleTimeString();
  } catch (e) {
    $("error").innerHTML = '<div class="err">' + esc(e.message) +
      ' &mdash; if authentication is enabled, paste an API key above.</div>';
    $("status").textContent = "unreachable";
    $("status").className = "pill bad";
  }
}
refresh();
timer = setInterval(refresh, 2000);
</script>
</body>
</html>
)HTML";
}

}  // namespace oracle
