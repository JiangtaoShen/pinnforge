/* PINNForge dashboard — vanilla JS, no build step, no third-party code.
 *
 * Everything agents wrote (summaries, transcripts, file names) is inserted
 * with textContent: the page never parses agent text as HTML.
 */
"use strict";
(() => {
  // ───────────────────────────── text ─────────────────────────────
  const TEXT = {
    runs: "Runs", tasks: "Tasks", compute: "Compute", noRuns: "No runs yet.",
    startHint: "Start one:", noTasks: "No tasks yet.", live: "live", idle: "idle",
    connecting: "connecting…", connected: "live", best: "Best rRMSE", anchor: "Anchor (b00)",
    vsAnchor: "{0}× better than the anchor", betterBy: "×{0} better", evals: "Evaluations", errors: "{0} failed",
    gpuTime: "GPU time", cpuTime: "CPU time", cost: "Model cost", tokens: "{0} tokens", progress: "Convergence",
    yTitle: "Relative L² error", noGpuRun: "no GPU in this run",
    metaWorkspace: "workspace", metaTask: "task", metaTrain: "train wall", metaAgent: "agent", metaModel: "model",
    legendEval: "evaluation", legendCur: "current block", legendBest: "best so far",
    legendErr: "failed", field: "Predicted field",
    noFrames: "The field appears here once an evaluation is scored.",
    play: "Play", pause: "Pause", timeSlice: "t =", agent: "Agent activity",
    noActivity: "No agent activity yet.", blocks: "Blocks", evalTable: "Evaluations",
    block: "Block", candidate: "Candidate", score: "rRMSE", wall: "Wall s", when: "When",
    status: "Status", spentOf: "{0} / {1} s", nEvals: "{0} evals", gpus: "GPUs", util: "Utilization",
    mem: "Memory", usedBy: "in use by {0}", free: "free", noGpu: "No GPU visible — CPU mode.",
    model: "Model", task: "Task", title: "Title", state: "State", ready: "ready",
    authoring: "authoring: {0}", failedAt: "failed at {0}", stages: "Authoring",
    reference: "Reference", refBuilt: "built by an agent", refOperator: "from the operator",
    estimate: "Estimated error", order: "Observed order",
    reproduced: "Reproduced", yes: "yes", no: "no", tolerance: "tolerance",
    levelDiff: "level vs next level", finestEst: "estimate for the finest",
    authorAgent: "Author agent", problem: "problem.md", runIt: "Run it:",
    notFound: "Not found.", workspace: "Workspace", back: "Overview", tFieldAt: "at",
    stage_intake: "Intake", stage_author: "Author the package", stage_reference: "Reference",
    stage_reference_check: "Check the reference", stage_validate: "Validate",
    stage_smoke: "Smoke test", stage_anchor: "Measure the anchor", stage_finalize: "Ready",
    st_done: "done", st_running: "running", st_failed: "failed", st_short: "short",
    st_interrupted: "interrupted", st_stopped: "stopped", st_finished: "finished",
    st_created: "created", st_pending: "pending", st_skipped: "skipped", st_ready: "ready",
    st_measuring: "measuring", st_missing: "missing", st_unreadable: "unreadable",
    st_clean: "clean", st_violated: "violated", st_unchecked: "unchecked",
    integrity: "integrity", blocksOf: "{0}/{1} blocks", table: "Table", snapshot: "snapshot",
    replay: "Discovery replay",
  };
  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const T = (key, ...args) => {
    let text = TEXT[key] ?? key;
    args.forEach((a, i) => { text = text.replace(`{${i}}`, a); });
    return text;
  };
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FILM = !!window.__PF_FILM__;  // this page is a discovery film, not the live dashboard

  // ───────────────────────────── DOM helpers ─────────────────────────────
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "style") Object.assign(el.style, v);
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const NS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null) continue;
      if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid) el.append(kid);
    return el;
  }
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // A panel's mark: a 16-pixel line drawing in the accent, with one small
  // motion that says what the panel holds (app.css, "panel icons").
  const ICONS = {
    agent: [["rect", { x: 2.5, y: 5, width: 11, height: 8.5, rx: 2 }], ["path", { d: "M8 2.8V5" }],
      ["circle", { class: "fill pulse", cx: 8, cy: 2.2, r: 1.1 }],
      ["g", { class: "blink" }, [["circle", { class: "fill", cx: 6, cy: 9.2, r: 1 }],
        ["circle", { class: "fill", cx: 10, cy: 9.2, r: 1 }]]]],
    field: [["rect", { x: 1.5, y: 1.5, width: 13, height: 13, rx: 1.5 }],
      ["path", { class: "flow", d: "M1.5 9.5C4 5.5 6 5.5 8 8.5S12 11.5 14.5 7" }]],
    convergence: [["path", { class: "dim", d: "M2 1.5V14.5H14.5" }],
      ["path", { class: "draw", pathLength: 1, d: "M3.8 3.5C5.5 9 8 11 13.5 12" }]],
    blocks: [["rect", { class: "step", x: 1.2, y: 5.6, width: 3.8, height: 4.8, rx: 0.6 }],
      ["rect", { class: "step", x: 6.1, y: 5.6, width: 3.8, height: 4.8, rx: 0.6 }],
      ["rect", { class: "step", x: 11, y: 5.6, width: 3.8, height: 4.8, rx: 0.6 }]],
    evals: [["path", { d: "M2 12A6 6 0 0 1 14 12" }], ["path", { class: "needle", d: "M8 12V6.6" }],
      ["circle", { class: "fill", cx: 8, cy: 12, r: 1.2 }]],
    compute: [["rect", { x: 4, y: 4, width: 8, height: 8, rx: 1 }],
      ["rect", { class: "fill pulse", x: 6.4, y: 6.4, width: 3.2, height: 3.2 }],
      ["path", { d: "M6 1.5V4M10 1.5V4M6 12V14.5M10 12V14.5M1.5 6H4M1.5 10H4M12 6H14.5M12 10H14.5" }]],
    runs: [["circle", { class: "spin", cx: 8, cy: 8, r: 6.2, "stroke-dasharray": "3 2.2" }],
      ["path", { class: "fill", d: "M6.6 5.4L10.8 8L6.6 10.6Z" }]],
    tasks: [["circle", { class: "step", cx: 3, cy: 4, r: 1.1 }], ["path", { d: "M6 4H14" }],
      ["circle", { class: "step", cx: 3, cy: 8, r: 1.1 }], ["path", { d: "M6 8H14" }],
      ["circle", { class: "step", cx: 3, cy: 12, r: 1.1 }], ["path", { d: "M6 12H11" }]],
    authoring: [["path", { d: "M10.5 2L13.5 5L6 12.5H3V9.5Z" }],
      ["path", { class: "draw", pathLength: 1, d: "M2 14.8H14" }]],
    reference: [["circle", { cx: 8, cy: 8, r: 5 }], ["circle", { class: "ring", cx: 8, cy: 8, r: 5 }],
      ["circle", { class: "fill", cx: 8, cy: 8, r: 1.8 }]],
    problem: [["path", { d: "M3.5 1.5H10L12.5 4V14.5H3.5Z" }], ["path", { d: "M5.8 6.5H10.2" }],
      ["path", { class: "draw", pathLength: 1, d: "M5.8 9.5H10.2M5.8 12H8.6" }]],
  };
  const mark = ([tag, attrs, kids]) => s(tag, attrs, ...(kids || []).map(mark));
  const icon = (name) => s("svg", { class: `ic ic-${name}`, viewBox: "0 0 16 16", "aria-hidden": "true" },
    ...(ICONS[name] || []).map(mark));

  // ───────────────────────────── formatting ─────────────────────────────
  function fmtScore(x) {
    if (x == null || Number.isNaN(x)) return "—";
    if (x === 0) return "0";
    const a = Math.abs(x);
    if (a >= 1e-3 && a < 1e4) return String(Number(x.toPrecision(4)));
    return x.toExponential(2);
  }
  function fmtCompact(n) {
    if (n == null) return "—";
    if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(Math.round(n));
  }
  function fmtDur(sec) {
    if (sec == null) return "—";
    sec = Math.round(sec);
    const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
    if (hh) return `${hh}h ${String(mm).padStart(2, "0")}m`;
    if (mm) return `${mm}m ${String(ss).padStart(2, "0")}s`;
    return `${ss}s`;
  }
  function relTime(iso) {
    if (!iso) return "";
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return iso;
    const d = (Date.now() - t) / 1000;
    const fmt = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
    if (d < 60) return fmt.format(-Math.round(d), "second");
    if (d < 3600) return fmt.format(-Math.round(d / 60), "minute");
    if (d < 86400) return fmt.format(-Math.round(d / 3600), "hour");
    return fmt.format(-Math.round(d / 86400), "day");
  }

  function statusEl(status) {
    const st = String(status || "pending");
    const ico = { done: "✓", finished: "✓", ready: "✓", clean: "✓", running: "●", failed: "✕",
      violated: "!", short: "~", interrupted: "‖", stopped: "■", pending: "·", created: "·",
      measuring: "●", missing: "?", unreadable: "!" }[st] || "·";
    return h("span", { class: `status ${st}` }, h("span", { class: "ico", text: ico }),
      h("span", { text: T("st_" + st) }));
  }

  // animated number: tweens from the old value shown to the new one
  function tween(el, to, fmt) {
    const from = Number(el.dataset.v);
    el.dataset.v = String(to);
    if (to == null || Number.isNaN(to)) { el.textContent = "—"; return; }
    if (reduced || Number.isNaN(from) || from === to) { el.textContent = fmt(to); return; }
    const t0 = performance.now(), dur = 700;
    const log = from > 0 && to > 0;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      const v = log ? Math.exp(Math.log(from) + (Math.log(to) - Math.log(from)) * e)
                    : from + (to - from) * e;
      el.textContent = fmt(k < 1 ? v : to);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ───────────────────────────── tooltip ─────────────────────────────
  const tip = document.getElementById("tooltip");
  let tipOwner = null;  // the element whose hover put the tooltip up
  function showTip(evt, rows) {
    tipOwner = evt.currentTarget instanceof Element ? evt.currentTarget : null;
    tip.replaceChildren(...rows);
    tip.hidden = false;
    const pad = 14, w = tip.offsetWidth, hgt = tip.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + hgt > window.innerHeight - 8) y = evt.clientY - hgt - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const hideTip = () => { tip.hidden = true; tipOwner = null; };
  // An owner redrawn under the pointer (a live refresh replaces it) never
  // hears the pointer leave, so the tooltip also goes when the pointer is over
  // anything else, or its owner has left the page.
  document.addEventListener("pointermove", (e) => {
    if (!tip.hidden && (!tipOwner || !tipOwner.isConnected || !tipOwner.contains(e.target))) hideTip();
  });
  document.documentElement.addEventListener("pointerleave", hideTip);  // out of the window
  const dropOrphanTip = () => { if (!tip.hidden && (!tipOwner || !tipOwner.isConnected)) hideTip(); };

  // ───────────────────────────── data ─────────────────────────────
  async function get(path) {
    const r = await fetch(path, { credentials: "same-origin", cache: "no-store" });
    if (!r.ok) {
      const err = new Error(`${r.status}`);
      err.status = r.status;
      throw err;
    }
    return r.json();
  }

  const S = {
    route: [], overview: null, gpus: [], gpuHist: {}, frames: new Map(),
    run: null, runSeen: {}, player: null, feed: null, selBlock: null, task: null, replay: null,
    ws: { name: "", path: "" },  // the workspace served, from /api/ping
  };

  // ───────────────────────────── routing ─────────────────────────────
  const app = document.getElementById("app");
  function parseRoute() {
    const parts = (location.hash.replace(/^#\/?/, "") || "").split("/").filter(Boolean)
      .map(decodeURIComponent);
    return parts;
  }
  async function route() {
    S.route = parseRoute();
    S.player && S.player.stop();
    S.player = null;
    S.replay && S.replay.destroy();
    S.replay = null;
    S.feed = null;
    S.selBlock = null;
    app.replaceChildren();
    crumbs();
    await refresh(true);
  }
  function crumbs() {
    const c = document.getElementById("crumbs");
    const [kind, id, sub] = S.route;
    const items = [h("a", { href: "#/", text: T("back") })];
    if (kind === "run" && id && sub === "replay" && FILM) {
      items.push(h("span", { class: "sep", text: "/" }), h("a", { href: `#/run/${encodeURIComponent(id)}`, text: id }),
        h("span", { class: "sep", text: "/" }), h("span", { text: T("replay") }));
    } else if (kind === "run" && id) items.push(h("span", { class: "sep", text: "/" }), h("span", { text: id }));
    if (kind === "task" && id) items.push(h("span", { class: "sep", text: "/" }), h("span", { text: id }));
    c.replaceChildren(...items);
  }

  let refreshing = false, again = false;
  async function refresh(first = false) {
    if (refreshing) { again = true; return; }
    refreshing = true;
    try {
      const [kind, id, sub] = S.route;
      if (kind === "run" && id && sub === "replay" && FILM) await viewReplay(id, first);
      else if (kind === "run" && id) await viewRun(id, first);
      else if (kind === "task" && id) await viewTask(id, first);
      else await viewOverview(first);
    } catch (e) {
      if (e.status === 404) app.replaceChildren(h("div", { class: "card empty", text: T("notFound") }));
      else console.error(e);
    } finally {
      refreshing = false;
      dropOrphanTip();
      if (again) { again = false; refresh(); }
    }
  }

  let timer = null;
  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(() => refresh(), 250);
  }

  // ───────────────────────────── live stream ─────────────────────────────
  const conn = document.getElementById("conn");
  function connect() {
    if (window.__PF_STATIC__) { conn.textContent = T("snapshot"); return; }
    conn.textContent = T("connecting");
    conn.classList.remove("live");
    let es;
    try { es = new EventSource("/api/events"); } catch { setTimeout(connect, 4000); return; }
    es.onopen = () => { conn.textContent = T("connected"); conn.classList.add("live"); };
    es.addEventListener("change", () => schedule());
    es.addEventListener("gpu", (e) => {
      try { S.gpus = JSON.parse(e.data).gpus || []; } catch { return; }
      for (const g of S.gpus) {
        const hist = S.gpuHist[g.index] || (S.gpuHist[g.index] = []);
        hist.push(g.util ?? 0);
        if (hist.length > 90) hist.shift();
      }
      updateGpus();
    });
    es.onerror = () => {
      conn.textContent = T("connecting");
      conn.classList.remove("live");
      // a stream the server ends is retried by EventSource itself; one it
      // refused outright (busy, restarted) is closed for good — start afresh
      if (es.readyState === EventSource.CLOSED) setTimeout(connect, 4000);
    };
  }

  // ───────────────────────────── overview ─────────────────────────────
  async function viewOverview(first) {
    const data = await get("/api/overview");
    S.overview = data;
    document.title = `PINNForge — ${data.name}`;
    if (first || !app.querySelector("#ov-runs")) {
      app.replaceChildren(
        h("div", { class: "page-title" }, h("h1", { text: data.name }),
          h("span", { class: "meta", text: `${data.workspace} · pinnforge ${data.version}` })),
        h("section", { class: "card" }, h("h2", {}, icon("compute"), T("compute")), h("div", { id: "ov-gpus", class: "gpus" })),
        h("section", {}, h("h2", { class: "section-h" }, icon("runs"), T("runs")), h("div", { id: "ov-runs", class: "grid-3" })),
        h("section", { class: "card" }, h("h2", {}, icon("tasks"), T("tasks")), h("div", { id: "ov-tasks" })),
      );
    }
    renderRunCards(document.getElementById("ov-runs"), data.runs);
    renderTaskTable(document.getElementById("ov-tasks"), data.tasks);
    if (!S.gpus.length) {
      try { S.gpus = (await get("/api/gpus")).gpus; } catch { /* no nvidia-smi */ }
    }
    updateGpus();
  }

  function renderRunCards(host, runs) {
    if (!runs.length) {
      host.replaceChildren(h("div", { class: "card empty" }, T("noRuns"), " ", T("startHint"),
        h("div", { class: "hint", text: "pinnforge demo\npinnforge run start -t demo -n 2" })));
      return;
    }
    const seen = new Set();
    for (const r of runs) {
      seen.add(r.id);
      let card = host.querySelector(`[data-run="${CSS.escape(r.id)}"]`);
      if (!card) {
        card = h("a", { class: "card run-card", href: `#/run/${encodeURIComponent(r.id)}`, "data-run": r.id },
          h("div", { class: "top" }, h("span", { class: "name", text: r.id }), h("span", { class: "task" }),
            h("span", { class: "st" })),
          h("div", { class: "nums" }, h("span", { class: "best", "data-v": "" }), h("span", { class: "vs" })),
          h("div", { class: "sparkh" }),
          h("div", { class: "foot" }, h("span", { class: "blk" }), h("span", { class: "upd" })));
        host.append(card);
      }
      card.querySelector(".task").textContent = r.task;
      card.querySelector(".st").replaceChildren(statusEl(r.live ? "running" : r.status));
      tween(card.querySelector(".best"), r.best ?? r.anchor, fmtScore);
      const ratio = r.best && r.anchor ? r.anchor / r.best : null;
      card.querySelector(".vs").textContent = ratio && ratio > 1.0001
        ? T("vsAnchor", ratio.toFixed(ratio >= 10 ? 0 : 1))
        : `${T("anchor")} ${fmtScore(r.anchor)}`;
      card.querySelector(".sparkh").replaceChildren(sparkline(r.trend || []));
      card.querySelector(".blk").textContent = T("blocksOf", r.blocks_done, r.blocks_total);
      card.querySelector(".upd").textContent = relTime(r.updated);
    }
    for (const card of [...host.children]) if (card.dataset.run && !seen.has(card.dataset.run)) card.remove();
    // live runs first, keep DOM order stable otherwise
    const live = runs.filter((r) => r.live).map((r) => r.id);
    for (const id of live.reverse()) {
      const c = host.querySelector(`[data-run="${CSS.escape(id)}"]`);
      if (c && host.firstChild !== c) host.prepend(c);
    }
  }

  function sparkline(values) {
    const W = 260, H = 36, P = 3;
    const svg = s("svg", { class: "spark", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", "aria-hidden": "true" });
    const vals = values.filter((v) => v > 0);
    if (vals.length < 2) return svg;
    const lo = Math.log(Math.min(...vals)), hi = Math.log(Math.max(...vals));
    const y = (v) => P + (hi === lo ? 0.5 : (hi - Math.log(v)) / (hi - lo)) * (H - 2 * P);
    const x = (i) => P + (i / (vals.length - 1)) * (W - 2 * P);
    const pts = vals.map((v, i) => [x(i), y(v)]);
    const line = pts.map(([a, b], i) => `${i ? "L" : "M"}${a.toFixed(1)},${b.toFixed(1)}`).join("");
    svg.append(s("path", { class: "area", d: `${line}L${x(vals.length - 1)},${H}L${x(0)},${H}Z` }),
      s("path", { class: "line", d: line }));
    const [lx, ly] = pts[pts.length - 1];
    svg.append(s("circle", { cx: lx, cy: ly, r: 3.5 }));
    return svg;
  }

  function renderTaskTable(host, tasks) {
    if (!tasks.length) {
      host.replaceChildren(h("div", { class: "empty" }, T("noTasks"),
        h("div", { class: "hint", text: "pinnforge task new" })));
      return;
    }
    const rows = tasks.map((t) => {
      let state;
      if (t.ready) state = statusEl("ready");
      else if (t.authoring && t.authoring.status === "failed") state = statusEl("failed");
      else state = statusEl("running");
      const label = t.ready ? null : (t.authoring && t.authoring.status === "failed"
        ? T("failedAt", T("stage_" + t.authoring.stage)) : T("authoring", T("stage_" + (t.authoring || {}).stage)));
      return h("tr", { class: "clickable", onclick: () => { location.hash = `#/task/${encodeURIComponent(t.name)}`; } },
        h("td", {}, h("b", { text: t.name })),
        h("td", { text: t.title }),
        h("td", {}, state, label ? h("span", { class: "pill", text: label }) : null),
        h("td", { class: "num", text: fmtScore(t.anchor) }));
    });
    host.replaceChildren(h("table", {},
      h("thead", {}, h("tr", {}, h("th", { text: T("task") }), h("th", { text: T("title") }),
        h("th", { text: T("state") }), h("th", { class: "num", text: T("anchor") }))),
      h("tbody", {}, rows)));
  }

  // ───────────────────────────── GPUs ─────────────────────────────
  function updateGpus() {
    const host = document.getElementById("ov-gpus") || document.getElementById("run-gpus");
    if (!host) return;
    if (!S.gpus.length) { host.replaceChildren(h("div", { class: "empty", text: T("noGpu") })); return; }
    const leases = (S.overview && S.overview.leases) || {};
    const pool = S.overview && S.overview.pool;
    for (const g of S.gpus) {
      let card = host.querySelector(`[data-gpu="${g.index}"]`);
      if (!card) {
        card = h("div", { class: "gpu", "data-gpu": String(g.index) },
          h("div", { class: "head" }, h("b", { text: `GPU ${g.index}` }), h("span", { class: "nm", text: g.name })),
          h("div", { class: "row" }, h("span", { text: T("util") }), h("div", { class: "meter util" }, h("i")), h("span", { class: "uv" })),
          h("div", { class: "row" }, h("span", { text: T("mem") }), h("div", { class: "meter mem" }, h("i")), h("span", { class: "mv" })),
          h("div", { class: "histh" }),
          h("div", { class: "who" }));
        host.append(card);
      }
      const util = g.util ?? 0, mem = g.mem_total ? (100 * (g.mem_used || 0)) / g.mem_total : 0;
      card.querySelector(".util > i").style.width = `${util}%`;
      card.querySelector(".mem > i").style.width = `${mem}%`;
      card.querySelector(".util").classList.toggle("hot", util >= 90);
      card.querySelector(".uv").textContent = `${Math.round(util)}%`;
      card.querySelector(".mv").textContent = `${((g.mem_used || 0) / 1024).toFixed(1)} GB`;
      card.querySelector(".histh").replaceChildren(utilHistory(S.gpuHist[g.index] || []));
      const holder = leases[g.index] ?? leases[String(g.index)];
      const unused = Array.isArray(pool) && !pool.includes(g.index);
      card.classList.toggle("unused", unused);  // dimmed: outside this workspace's cards
      const parts = [
        holder ? h("a", { href: `#/run/${encodeURIComponent(holder)}`, text: T("usedBy", holder) })
          : unused ? null : h("span", { text: T("free") }),
        g.temp != null ? `${Math.round(g.temp)} °C` : null,
        g.power != null ? `${Math.round(g.power)} W` : null,
      ].filter(Boolean);
      card.querySelector(".who").replaceChildren(...parts.flatMap((p, i) => (i ? [" · ", p] : [p])));
    }
  }
  function utilHistory(hist) {
    const W = 220, H = 34;
    const svg = s("svg", { class: "gauge", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", "aria-hidden": "true" });
    if (hist.length < 2) return svg;
    const x = (i) => W - (hist.length - 1 - i) * (W / 89);
    const y = (v) => H - 2 - (Math.min(100, v) / 100) * (H - 4);
    const line = hist.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    svg.append(s("path", { d: `${line}L${x(hist.length - 1)},${H}L${x(0)},${H}Z`, fill: css("--accent-wash") }),
      s("path", { d: line, fill: "none", stroke: css("--accent"), "stroke-width": 1.5 }));
    return svg;
  }

  // ───────────────────────────── run view ─────────────────────────────
  async function viewRun(id, first) {
    const run = await get(`/api/runs/${encodeURIComponent(id)}`);
    const prev = S.run && S.run.id === id ? S.run : null;
    S.run = run;
    document.title = `${run.id} — PINNForge`;
    if (first || !app.querySelector("#run-chart")) buildRunLayout(run);
    if (!S.selBlock) {
      const blocks = run.blocks.filter((b) => b.id !== "b00");
      const running = blocks.filter((b) => b.status === "running");
      S.selBlock = (running[running.length - 1] || blocks[blocks.length - 1] || {}).id || null;
    }
    updateRunHeader(run);
    updateTiles(run, prev);
    fitScreen();  // the tiles above it are in place now
    renderGantt(document.getElementById("run-gantt"), run);  // its rows take room from the chart
    renderScoreChart(document.getElementById("run-chart"), run, prev);
    await Promise.all([updatePlayer(run), updateFeed(run)]);
    renderEvalTable(document.getElementById("run-table"), run);
  }

  // A finished run as a film (replay.js), in the file an agent's script was
  // rendered into (pinnforge/discovery.py): mounted once, never refreshed.
  async function viewReplay(id, first) {
    if (!first && S.replay) return;
    document.title = `${id} — ${T("replay")} — PINNForge`;
    const host = h("div", { class: "replay-page" });
    app.replaceChildren(host);
    S.replay = await window.PFReplay.mount(host, id);
  }

  // a run with no card trains on the CPU; its clock is not a GPU's
  const onGpu = (run) => ((run.config && run.config.gpus) || []).length > 0;
  const timeLabel = (run) => T(onGpu(run) ? "gpuTime" : "cpuTime");

  // A run on one screen, in three columns: the agent, given a whole column
  // for the most text; the score over the blocks; the evaluations over the
  // field they produced.
  function buildRunLayout(run) {
    const card = (cls, ic, title, asideId, ...body) => h("section", { class: `card ${cls}` },
      h("h2", {}, icon(ic), title, asideId ? h("span", { class: "aside", id: asideId }) : null), ...body);
    // the field's player controls ride in its header, leaving the panel to the field
    const fieldCard = card("c-field", "field", T("field"), null,
      h("div", { id: "run-field", class: "field" }));
    fieldCard.firstChild.append(h("span", { class: "head-tools", id: "field-tools", hidden: true }));
    app.replaceChildren(
      h("div", { class: "page-title" }, h("h1", { id: "run-title" }),
        h("span", { class: "meta", id: "run-meta" }), h("span", { id: "run-status" })),
      h("div", { class: "tiles run-tiles", id: "run-tiles" }),
      h("div", { class: "cockpit", id: "run-cockpit" },
        h("div", { class: "col" },
          card("c-agent", "agent", T("agent"), "feed-aside", h("div", { id: "run-feed", class: "feed", "aria-live": "polite" }))),
        h("div", { class: "col" },
          card("c-progress", "convergence", T("progress"), null, h("div", { id: "run-chart" }),
            h("div", { class: "legend" },
              h("span", {}, h("i", { class: "k-dot" }), T("legendEval")),
              h("span", {}, h("i", { class: "k-cur" }), T("legendCur")),
              h("span", {}, h("i", { class: "k-line" }), T("legendBest")),
              h("span", {}, h("i", { class: "k-err" }), T("legendErr")))),
          card("c-blocks", "blocks", T("blocks"), null, h("div", { id: "run-gantt", class: "gantt" }))),
        h("div", { class: "col" },
          card("c-evals", "evals", T("evalTable"), "evals-aside", h("div", { id: "run-table", class: "table-box" })),
          fieldCard)),
    );
  }

  // The one-screen grids take the window's height below where they start.
  function fitScreen() {
    for (const el of app.querySelectorAll(".cockpit, .screen")) {
      if (getComputedStyle(el).display !== "grid") { el.style.height = ""; continue; }
      const top = el.getBoundingClientRect().top + window.scrollY;
      el.style.height = `${Math.max(460, window.innerHeight - top - 28)}px`;  // main's own foot
    }
  }
  // inside a fitted grid a chart takes the height it is given
  const fitted = (el) => {
    const grid = el.closest(".cockpit, .screen");
    return !!grid && getComputedStyle(grid).display === "grid";
  };

  // the agents a run can be given, by the id its run.yaml names
  const RUNTIMES = {
    replay: ["Scripted demo agent", "replays a fixed playbook; calls no model"],
    claude_code: ["Claude Code"], codex: ["Codex"], cursor_agent: ["Cursor Agent"], opencode: ["OpenCode"],
  };
  // hovering an element shows its rows ([label, text, isPath]) in the tooltip
  function hoverTip(el, rows) {
    rows = rows.filter((r) => r && r[1]);
    if (!rows.length) return el;
    const draw = () => rows.map(([k, v, path]) => h("div", { class: "tip-row" },
      h("span", { class: "tip-k", text: k }), h("span", { class: path ? "tip-p" : "tip-v", text: v })));
    el.addEventListener("pointermove", (e) => showTip(e, draw()));
    el.addEventListener("pointerleave", hideTip);
    el.setAttribute("aria-label", rows.map(([k, v]) => `${k}: ${v}`).join("; "));
    return el;
  }
  function updateRunHeader(run) {
    const cfg = run.config || {};
    const agent = cfg.agent || {};
    const title = document.getElementById("run-title");
    title.textContent = run.id;
    if (!title.dataset.tip) {
      title.dataset.tip = "1";
      hoverTip(title, [["Run", run.id], ["Path", cfg.path, true]]);
    }
    const kv = (k, v, rows) => (v ? hoverTip(h("span", { class: "kv" }, h("span", { class: "k", text: T(k) }),
      h("span", { class: "v", text: v })), rows) : null);
    const [name, about] = RUNTIMES[agent.runtime] || [agent.runtime];
    const used = [...new Set(run.blocks.map((b) => b.model).filter(Boolean))].join(", ");
    const model = ["Model", agent.model || "the agent's default"];
    // rebuilt only when what they say changes, so a chip under the pointer stays put
    const meta = document.getElementById("run-meta");
    const train = cfg.train_s ? (cfg.train_s >= 600 ? fmtDur(cfg.train_s) : `${Math.round(cfg.train_s)} s`) : "";
    const key = JSON.stringify([S.ws, run.task, cfg.task_title, cfg.task_path, train, agent, used]);
    if (meta.dataset.key !== key) {
      meta.dataset.key = key;
      meta.replaceChildren(...[
        kv("metaWorkspace", S.ws.name, [["Workspace", S.ws.name], ["Path", S.ws.path, true]]),
        kv("metaTask", run.task, [["Task", run.task], ["Problem", cfg.task_title], ["Path", cfg.task_path, true]]),
        kv("metaTrain", train, [["Train wall", `${train} of wall clock per evaluation`],
          ["", "each candidate trains this long at most; fixed by the task (TRAIN_TIME)"]]),
        kv("metaAgent", agent.runtime, [["Agent", name], ["", about], model, ["Models used", used]]),
        kv("metaModel", agent.model, [model, ["Models used", used]])].filter(Boolean));
    }
    document.getElementById("run-status").replaceChildren(statusEl(run.live ? "running" : run.status));
  }

  function tile(id, label) {
    return h("div", { class: "tile", id }, h("div", { class: "label", text: label }),
      h("div", { class: "value", "data-v": "" }), h("div", { class: "note" }));
  }
  function updateTiles(run, prev) {
    const host = document.getElementById("run-tiles");
    if (!host.children.length) {
      host.append(tile("t-best", T("best")), tile("t-anchor", T("anchor")), tile("t-evals", T("evals")),
        tile("t-gpu", T("gpuTime")), tile("t-cpu", T("cpuTime")), tile("t-cost", T("cost")));
      host.firstChild.classList.add("hero");
    }
    const q = (id, sel) => document.querySelector(`#${id} ${sel}`);
    tween(q("t-best", ".value"), run.best, fmtScore);
    const ratio = run.best && run.anchor ? run.anchor / run.best : null;
    q("t-best", ".note").replaceChildren(ratio && ratio > 1.0001
      ? h("span", { class: "delta-good", text: T("betterBy", ratio.toFixed(ratio >= 10 ? 0 : 1)),
        title: T("vsAnchor", ratio.toFixed(ratio >= 10 ? 0 : 1)) })
      : h("span", { text: run.best_block ? run.best_block : "" }));
    if (prev && run.best != null && prev.best != null && run.best < prev.best && !reduced) {
      const t = document.getElementById("t-best");
      t.classList.remove("flash"); void t.offsetWidth; t.classList.add("flash");
    }
    tween(q("t-anchor", ".value"), run.anchor, fmtScore);
    const scored = run.evals.filter((e) => !e.diag);
    const failed = scored.filter((e) => e.rRMSE == null).length;
    tween(q("t-evals", ".value"), scored.length, (v) => String(Math.round(v)));
    q("t-evals", ".note").textContent = failed ? T("errors", failed) : "";
    // The run's own clock (what its budget is charged in) for the device it
    // trains on; the other device's time from the evaluations that ran there.
    const spent = run.blocks.filter((b) => b.id !== "b00").reduce((a, b) => a + (b.spent || 0), 0);
    const budget = (run.config.wall_budget_s || 0) * (run.blocks_total || 0);
    const onCard = (e) => typeof e.gpu === "number";
    const elsewhere = scored.filter((e) => e.block !== "b00" && onCard(e) !== onGpu(run))
      .reduce((a, e) => a + (e.wall_s || 0), 0);
    const [gpuS, cpuS] = onGpu(run) ? [spent, elsewhere] : [elsewhere, spent];
    tween(q("t-gpu", ".value"), onGpu(run) || gpuS ? gpuS : null, fmtDur);
    q("t-gpu", ".note").textContent = onGpu(run) ? (budget ? `/ ${fmtDur(budget)}` : "") : (gpuS ? "" : T("noGpuRun"));
    tween(q("t-cpu", ".value"), cpuS, fmtDur);
    q("t-cpu", ".note").textContent = !onGpu(run) && budget ? `/ ${fmtDur(budget)}` : "";
    const cost = run.blocks.reduce((a, b) => a + (b.cost_usd || 0), 0);
    const tokens = run.blocks.reduce((a, b) => a + (b.tokens || 0), 0);
    tween(q("t-cost", ".value"), cost, (v) => (cost ? `$${v.toFixed(2)}` : "—"));
    q("t-cost", ".note").textContent = tokens ? T("tokens", fmtCompact(tokens)) : "";
  }

  // ── score chart ──
  function renderScoreChart(host, run, prev) {
    // in the one-screen grid the chart takes exactly the room it is given
    const fit = fitted(host);
    const W = Math.max(fit ? 200 : 320, host.clientWidth || 800);
    const H = fit ? Math.max(100, host.clientHeight || 300) : 300;
    const evs = run.evals.filter((e) => !e.diag);
    const pts = evs.filter((e) => e.rRMSE != null && e.rRMSE > 0);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `${T("progress")}: ${pts.length} ${T("evals")}` });
    const maxX = Math.max(1, ...evs.map((e) => e.cum_wall || 0));
    const vals = pts.map((p) => p.rRMSE).concat(run.anchor ? [run.anchor] : []);
    if (!vals.length) {
      host.replaceChildren(h("div", { class: "empty", text: T("noFrames") }));
      return;
    }
    let lo = Math.log10(Math.min(...vals)), hi = Math.log10(Math.max(...vals));
    lo = Math.floor((lo - 0.08) * 2) / 2; hi = Math.ceil((hi + 0.04) * 2) / 2;
    if (hi - lo < 0.5) { lo -= 0.25; hi += 0.25; }
    // ticks at decades and half-decades; the left margin fits the widest
    // label and, beside it, the y-axis title turned on its side
    const charW = 7.2;  // about 0.6em at the small type size
    const ticks = [];
    for (let d = Math.ceil(lo * 2) / 2; d <= hi + 1e-9; d += 0.5) {
      const v = Math.pow(10, d);
      ticks.push([v, fmtScore(Number(v.toPrecision(2)))]);
    }
    const m = { l: 24 + Math.max(4, ...ticks.map(([, t]) => t.length)) * charW + 8, r: 16, t: 22, b: 34 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const x = (v) => m.l + (v / maxX) * iw;
    const y = (v) => m.t + ((hi - Math.log10(v)) / (hi - lo)) * ih;

    // block bands
    const blockSpans = [];
    for (const e of evs) {
      const last = blockSpans[blockSpans.length - 1];
      if (!last || last.block !== e.block) blockSpans.push({ block: e.block, x0: (e.cum_wall || 0) - (e.wall_s || 0), x1: e.cum_wall || 0 });
      else last.x1 = e.cum_wall || 0;
    }
    // a band's label sits over it, or is left out when the band has no room
    blockSpans.forEach((b, i) => {
      const x0 = x(Math.max(0, b.x0)), x1 = x(b.x1);
      if (i % 2 === 1) svg.append(s("rect", { class: "band", x: x0, y: m.t, width: Math.max(0, x1 - x0), height: ih }));
      if (x1 - x0 > 26 && x1 - x0 >= b.block.length * charW) svg.append(s("text", { class: "band-label", x: (x0 + x1) / 2, y: m.t - 8, "text-anchor": "middle", text: b.block }));
    });
    for (const [v, label] of ticks) {
      const yy = y(v);
      svg.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: yy, y2: yy }),
        s("text", { class: "tick", x: m.l - 8, y: yy + 4, "text-anchor": "end", text: label }));
    }
    svg.append(s("line", { class: "axis", x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }));
    const step = niceStep(maxX / 6);
    for (let v = 0; v <= maxX + 1e-9; v += step) {
      svg.append(s("text", { class: "tick", x: x(v), y: H - m.b + 16, "text-anchor": "middle", text: fmtDur(v) }));
    }
    svg.append(s("text", { class: "axis-label", x: W - m.r, y: H - 4, "text-anchor": "end", text: `${timeLabel(run)} →` }));
    svg.append(s("text", { class: "axis-label", x: 0, y: 0, transform: `translate(12 ${m.t + ih / 2}) rotate(-90)`,
      "text-anchor": "middle", "dominant-baseline": "central", text: T("yTitle") }));
    // anchor
    if (run.anchor) {
      const ya = y(run.anchor);
      svg.append(s("line", { class: "anchor-line", x1: m.l, x2: W - m.r, y1: ya, y2: ya }),
        s("text", { class: "anchor-label", x: W - m.r - 4, y: ya - 6, "text-anchor": "end", text: `b00 ${fmtScore(run.anchor)}` }));
    }
    // best-so-far staircase
    let best = run.anchor || null;
    const stair = [];
    for (const e of evs) {
      if (e.rRMSE == null || e.rRMSE <= 0) continue;
      if (best == null || e.rRMSE < best) {
        if (best != null) stair.push([x(e.cum_wall), y(best)]);
        best = e.rRMSE;
        stair.push([x(e.cum_wall), y(best)]);
      }
    }
    if (stair.length) {
      stair.unshift([m.l, stair[0][1]]);
      stair.push([x(maxX), stair[stair.length - 1][1]]);
      const d = stair.map(([a, b], i) => `${i ? "L" : "M"}${a.toFixed(1)},${b.toFixed(1)}`).join("");
      svg.append(s("path", { class: "best-area", d: `${d}L${x(maxX)},${m.t + ih}L${m.l},${m.t + ih}Z` }));
      const path = s("path", { class: "best", d });
      svg.append(path);
      const prevBest = prev ? prev.best : undefined;
      if (!reduced && (!prev || (run.best != null && run.best !== prevBest))) {
        requestAnimationFrame(() => {
          const len = path.getTotalLength();
          path.style.strokeDasharray = `${len}`;
          path.style.setProperty("--len", `${len}`);
          path.classList.add("draw");
        });
      }
    }
    // failed evaluations: ticks along the top
    for (const e of evs.filter((z) => z.rRMSE == null && z.cum_wall)) {
      svg.append(s("line", { class: "err-tick", x1: x(e.cum_wall), x2: x(e.cum_wall), y1: m.t + 2, y2: m.t + 10 }));
    }
    // points
    const seen = S.runSeen[run.id] || 0;
    const cur = S.selBlock;
    const hitLayer = [];
    pts.forEach((p, i) => {
      const cx = x(p.cum_wall), cy = y(p.rRMSE);
      const isNew = i >= seen && seen > 0 && !reduced;
      const dot = s("circle", { class: `pt${p.block === cur ? " cur" : ""}${isNew ? " enter" : ""}`, cx, cy, r: 4.5 });
      svg.append(dot);
      if (isNew) svg.append(s("circle", { class: "ripple", cx, cy, r: 6 }));
      hitLayer.push(s("circle", { class: "hit", cx, cy, r: 12,
        onpointermove: (evt) => showTip(evt, [
          h("div", { class: "v", text: fmtScore(p.rRMSE) }),
          h("div", { class: "r" }, h("span", { text: `${p.block} · ${p.candidate}` })),
          h("div", { class: "r", text: `${fmtDur(p.cum_wall)} · ${relTime(p.ts)}` })]),
        onpointerleave: hideTip,
        onclick: () => { S.selBlock = p.block; refresh(); } }));
    });
    svg.append(...hitLayer);
    S.runSeen[run.id] = pts.length;
    host.replaceChildren(svg);
  }
  function niceStep(raw) {
    const choices = [30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400];
    return choices.find((c) => c >= raw) || raw;
  }

  // ── blocks ──
  function renderGantt(host, run) {
    const blocks = run.blocks.filter((b) => b.id !== "b00");
    if (!blocks.length) { host.replaceChildren(h("div", { class: "empty", text: "—" })); return; }
    const seen = new Set();
    for (const b of blocks) {
      seen.add(b.id);
      let row = host.querySelector(`[data-b="${b.id}"]`);
      if (!row) {
        row = h("div", { class: "row", "data-b": b.id, role: "button", tabindex: "0",
          onclick: () => { S.selBlock = b.id; refresh(); },
          onkeydown: (e) => { if (e.key === "Enter") { S.selBlock = b.id; refresh(); } } },
          h("span", { class: "id", text: b.id }),
          h("div", { class: "bar" }, h("div", { class: "meter" }, h("i")), h("div", { class: "meta" })),
          h("div", { class: "right" }));
        host.append(row);
      }
      row.classList.toggle("sel", b.id === S.selBlock);
      const frac = b.budget ? Math.min(1, (b.spent || 0) / b.budget) : 0;
      const meter = row.querySelector(".meter");
      meter.querySelector("i").style.width = `${(100 * frac).toFixed(1)}%`;
      meter.classList.toggle("full", b.status === "done");
      const meta = row.querySelector(".meta");
      meta.replaceChildren(...[statusEl(b.status), h("span", { text: T("spentOf", Math.round(b.spent || 0), Math.round(b.budget || 0)) }),
        h("span", { text: [T("nEvals", b.evals), b.model, b.cost_usd ? `$${b.cost_usd.toFixed(2)}` : ""].filter(Boolean).join(" · ") }),
        b.integrity === "violated" ? statusEl("violated") : null].filter(Boolean));
      row.querySelector(".right").replaceChildren(h("b", { text: fmtScore(b.best) }),
        h("div", { text: b.completed ? relTime(b.completed) : (b.started ? relTime(b.started) : "") }));
    }
    for (const row of [...host.children]) if (row.dataset.b && !seen.has(row.dataset.b)) row.remove();
  }

  // ── predicted field ──
  // Turbo, the replay's map too: a diverging field gets a range centred on zero
  // (drawPanel), so zero sits at its middle either way.
  const TURBO = ["#30123B", "#4145AB", "#4675ED", "#39A2FC", "#1BCFD4", "#24ECA6", "#61FC6C", "#A4FC3B",
    "#D1E834", "#F3C63A", "#FE9B2D", "#F36315", "#D93807", "#B11901", "#7A0403"];
  function colormap() {
    const stops = TURBO.map((c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)));
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const f = (i / 255) * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(f)), u = f - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * u;
    }
    return lut;
  }

  // What the field window draws: the task's view.json when the run has a
  // valid one (checked server-side against the archived coordinates and
  // output heads), else a default worked out from the coordinates.
  function resolveView(run) {
    const grid = run.grid || {};
    const coords = (grid.coords || []).slice(0, 3);
    const fields = grid.fields && grid.fields.length ? grid.fields : ["u"];
    if (run.view && run.view.axes && run.view.panels && run.view.panels.length) return run.view;
    const isT = (c) => /^t(ime)?$/i.test(c);
    const view = { axes: {}, panels: [] };
    const t = coords.find(isT);
    if (coords.length === 1) view.axes = { x: coords[0] };
    else if (coords.length === 2) view.axes = t ? { x: coords.find((c) => c !== t), y: t } : { x: coords[0], y: coords[1] };
    else if (coords.length >= 3) {
      if (t) { const rest = coords.filter((c) => c !== t); view.axes = { x: rest[0], y: rest[1] }; view.time = t; }
      else { view.axes = { x: coords[0], y: coords[1] }; view.slice = { axis: coords[2], at: 0.5 }; }
    }
    view.panels = fields.slice(0, 4).map((f) => ({ title: f, show: { op: "field", of: [f] }, colors: "auto" }));
    return view;
  }

  // a panel's values, derived from the predicted heads only
  function panelValues(show, fields) {
    const cols = show.of.map((n) => fields[n] && fields[n].values);
    if (cols.some((c) => !c)) return null;
    if (show.op === "abs") return cols[0].map(Math.abs);
    if (show.op === "magnitude") return cols[0].map((_, i) => Math.hypot(...cols.map((c) => c[i])));
    return cols[0];
  }

  // Rows × columns, and each panel's width and height, that make n panels as
  // large as the window allows. A panel keeps the domain's aspect unless
  // `aspect` is null (a space-time map, whose axes share no unit): then it
  // takes whatever shape its cell has.
  function panelGrid(n, aspect, W, H, chrome) {
    if (!Number.isFinite(H)) {
      const cols = Math.max(1, Math.min(n, Math.floor(W / 300)));
      const w = (W - 12 * (cols - 1)) / cols;
      return { cols, rows: Math.ceil(n / cols), w, h: w * (aspect ?? 0.6) };
    }
    let best = null;
    for (let cols = 1; cols <= n; cols++) {
      const rows = Math.ceil(n / cols);
      const cw = (W - 12 * (cols - 1)) / cols;
      const ch = Math.max(30, (H - 12 * (rows - 1)) / rows - chrome);
      const w = aspect == null ? cw : Math.max(40, Math.min(cw, ch / aspect));
      const h = aspect == null ? ch : w * aspect;
      if (!best || w * h > best.w * best.h * 1.01) best = { cols, rows, w, h };
    }
    return best;
  }
  const timeLike = (c) => /^t(ime)?$/i.test(c || "");

  class Player {
    constructor(host, run) {
      this.host = host;
      this.runId = run.id;
      this.grid = run.grid || {};
      this.coords = (this.grid.coords && this.grid.coords.length ? this.grid.coords : ["x", "y"]).slice(0, 3);
      this.view = resolveView(run);
      this.frames = [];
      this.index = -1;
      this.step = null;
      this.follow = true;
      this.playing = false;
      this.panels = [];
      this.caption = h("div", { class: "caption" });
      this.grid_el = h("div", { class: "panels" });
      this.box = h("div", { class: "canvas-box" }, this.grid_el);
      this.slider = h("input", { type: "range", min: "0", max: "0", value: "0", "aria-label": T("field"),
        // dragged to the end, the slider follows new frames again
        oninput: () => { const i = Number(this.slider.value); this.follow = i === this.frames.length - 1; this.show(i); } });
      this.playBtn = h("button", { class: "btn", type: "button", title: T("play"), "aria-label": T("play"),
        onclick: () => this.toggle() }, "▶");
      this.stepSel = h("input", { type: "range", min: "0", max: "0", value: "0", hidden: true,
        oninput: () => { this.step = Number(this.stepSel.value); this.draw(); } });
      this.stepLabel = h("span", { class: "muted" });
      this.stepRow = h("div", { class: "controls step", hidden: true }, this.stepLabel, this.stepSel);
      this.tools = document.getElementById("field-tools") || h("div", { class: "controls" });
      this.tools.replaceChildren(this.playBtn, this.slider);
      host.replaceChildren(this.caption, this.box, ...(this.tools.isConnected ? [] : [this.tools]), this.stepRow);
    }
    stop() { clearInterval(this.timer); this.playing = false; }
    mode(next) {  // what the play button does next
      this.playBtn.textContent = next === "play" ? "▶" : "❚❚";
      this.playBtn.title = T(next);
      this.playBtn.setAttribute("aria-label", T(next));
    }
    toggle() {
      if (this.playing) { this.stop(); this.mode("play"); return; }
      this.playing = true;
      this.follow = false;
      this.mode("pause");
      let i = 0;
      this.show(0);
      this.timer = setInterval(() => {
        i += 1;
        if (i >= this.frames.length) { this.stop(); this.mode("play"); this.follow = true; return; }
        this.show(i);
      }, 900);
    }
    async update(frames) {
      const fresh = frames.filter((f) => f.run_best);
      const changed = fresh.length !== this.frames.length;
      this.frames = fresh;
      this.slider.max = String(Math.max(0, fresh.length - 1));
      this.tools.hidden = !fresh.length;
      this.host.classList.toggle("no-frames", !fresh.length);
      if (!fresh.length) {
        this.caption.replaceChildren(h("span", { class: "muted", text: T("noFrames") }));
        return;
      }
      if (changed && this.follow && !this.playing) await this.show(fresh.length - 1);
      else if (this.index < 0) await this.show(fresh.length - 1);
    }
    async load(n) {
      const key = `${this.runId}:${n}`;
      if (!S.frames.has(key)) S.frames.set(key, await get(`/api/runs/${encodeURIComponent(this.runId)}/frame/${n}`));
      return S.frames.get(key);
    }
    async show(i) {
      if (i < 0 || i >= this.frames.length) return;
      this.index = i;
      this.slider.value = String(i);
      const meta = this.frames[i];
      this.data = await this.load(meta.frame);
      this.caption.replaceChildren(h("b", { text: fmtScore(meta.rRMSE) }),
        h("span", { text: `${meta.block} · ${String(meta.candidate).split("/").pop()}` }),
        h("span", { class: "muted", text: `${i + 1} / ${this.frames.length}` }));
      if (!reduced) {
        this.grid_el.style.opacity = "0.35";
        requestAnimationFrame(() => { this.draw(); this.grid_el.style.opacity = "1"; });
      } else this.draw();
    }
    // which rows are on screen: all of them, or one value of the time/slice coordinate
    rows(cols) {
      const v = this.view;
      const name = v.time || (v.slice && v.slice.axis) || null;
      const si = name ? this.coords.indexOf(name) : -1;
      const all = cols[0].map((_, i) => i);
      this.stepRow.hidden = si < 0;
      this.stepSel.hidden = si < 0;
      if (si < 0) return all;
      const steps = [...new Set(cols[si])].sort((a, b) => a - b);
      if (this.step == null || this.step >= steps.length) {
        this.step = v.time ? steps.length - 1 : Math.round(((v.slice && v.slice.at) ?? 0.5) * (steps.length - 1));
      }
      this.stepSel.max = String(steps.length - 1);
      this.stepSel.value = String(this.step);
      const at = steps[this.step];
      this.stepLabel.textContent = `${name} = ${Number(at.toPrecision(4))}`;
      return all.filter((i) => cols[si][i] === at);
    }
    draw() {
      const d = this.data;
      if (!d) return;
      const v = this.view, cols = d.coords;
      const xi = this.coords.indexOf(v.axes.x);
      const yi = v.axes.y != null ? this.coords.indexOf(v.axes.y) : -1;
      if (xi < 0 || xi >= cols.length || (v.axes.y != null && (yi < 0 || yi >= cols.length))) {
        if (this.view !== this.fallback) { this.view = this.fallback = resolveView({ grid: this.grid }); this.draw(); }
        return;
      }
      const idx = this.rows(cols);
      const X = cols[xi], Y = yi >= 0 ? cols[yi] : null;
      const shown = v.panels.map((p) => ({ p, vals: panelValues(p.show, d.fields) })).filter((q) => q.vals);
      if (!shown.length) { this.grid_el.replaceChildren(h("div", { class: "empty", text: T("noFrames") })); return; }
      let aspect = 0.45;
      if (Y) {
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (let i = 0; i < X.length; i++) {
          if (X[i] < x0) x0 = X[i]; if (X[i] > x1) x1 = X[i];
          if (Y[i] < y0) y0 = Y[i]; if (Y[i] > y1) y1 = Y[i];
        }
        aspect = Math.min(1.6, Math.max(0.35, (y1 - y0) / Math.max(1e-12, x1 - x0)));
      }
      const fit = fitted(this.host);
      const W = this.box.clientWidth || this.host.clientWidth || 600;
      const H = fit ? this.box.clientHeight : Infinity;
      const spaceTime = !!Y && (timeLike(v.axes.y) || timeLike(v.axes.x));
      const lay = panelGrid(shown.length, spaceTime ? null : aspect, W, H, 46);
      this.grid_el.style.gridTemplateColumns = `repeat(${lay.cols}, ${Math.floor(lay.w)}px)`;
      while (this.panels.length < shown.length) {
        const canvas = h("canvas", { width: 10, height: 10 });
        const panel = { canvas, title: h("div", { class: "ptitle" }), bar: h("canvas", { width: 256, height: 8 }),
          lo: h("span"), hi: h("span") };
        panel.el = h("div", { class: "panel" }, panel.title, canvas, h("div", { class: "colorbar" }, panel.lo, panel.bar, panel.hi));
        canvas.addEventListener("pointermove", (e) => this.hover(e, panel));
        canvas.addEventListener("pointerleave", hideTip);
        this.panels.push(panel);
      }
      this.grid_el.replaceChildren(...this.panels.slice(0, shown.length).map((p) => p.el));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      shown.forEach((q, k) => this.drawPanel(this.panels[k], q, idx, X, Y, lay.w, lay.h, dpr));
    }
    drawPanel(panel, q, idx, X, Y, w, h, dpr) {
      const vals = q.vals;
      panel.title.textContent = q.p.title;
      panel.label = q.p.title;
      let lo = Infinity, hi = -Infinity;
      for (const i of idx) { const v = vals[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
      // diverging centres on zero; "auto" uses it only when both signs carry real weight
      const diverging = q.p.colors === "diverging" ||
        (q.p.colors === "auto" && lo < 0 && hi > 0 && Math.min(-lo, hi) >= 0.2 * Math.max(-lo, hi));
      if (diverging) { const a = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -a; hi = a; }
      if (hi === lo) hi = lo + 1;
      const lut = colormap();
      const canvas = panel.canvas;
      const Wp = Math.max(40, Math.round(w * dpr));
      const Hp = Math.max(30, Math.round(h * dpr));
      canvas.width = Wp; canvas.height = Hp;
      canvas.style.width = `${Math.round(Wp / dpr)}px`;
      canvas.style.height = `${Math.round(Hp / dpr)}px`;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, Wp, Hp);
      if (!Y) {
        panel.pick = null;
        drawLine(ctx, idx, X, vals, lo, hi, Wp, Hp, dpr);
        panel.bar.hidden = true;
      } else {
        panel.bar.hidden = false;
        panel.pick = drawTensor(ctx, idx, X, Y, vals, lo, hi, lut, Wp, Hp) || drawSplat(ctx, idx, X, Y, vals, lo, hi, lut, Wp, Hp);
        panel.pick.X = X; panel.pick.Y = Y; panel.pick.vals = vals;
        drawBar(panel.bar, lut);
      }
      panel.lo.textContent = fmtScore(lo); panel.hi.textContent = fmtScore(hi);
    }
    hover(evt, panel) {
      const pk = panel.pick;
      if (!pk) return;
      const r = panel.canvas.getBoundingClientRect();
      const px = Math.floor(((evt.clientX - r.left) / r.width) * pk.W);
      const py = Math.floor(((evt.clientY - r.top) / r.height) * pk.H);
      const i = pk.pick[py * pk.W + px];
      if (i == null || i < 0) { hideTip(); return; }
      showTip(evt, [h("div", { class: "v", text: `${panel.label} = ${fmtScore(pk.vals[i])}` }),
        h("div", { text: `${this.view.axes.x} ${Number(pk.X[i].toPrecision(4))} · ${this.view.axes.y} ${Number(pk.Y[i].toPrecision(4))}` })]);
    }
  }

  // Points on a lattice render as an image of its own cells, scaled up with
  // smoothing: exact, and far crisper than point splats. A lattice with holes —
  // where a cylinder or an airfoil sits — leaves those cells empty. Returns the
  // hover map, or null when the points are not on a lattice.
  function drawTensor(ctx, idx, X, Y, vals, lo, hi, lut, W, H) {
    const ux = [...new Set(idx.map((i) => X[i]))].sort((a, b) => a - b);
    const uy = [...new Set(idx.map((i) => Y[i]))].sort((a, b) => a - b);
    const nx = ux.length, ny = uy.length;
    if (nx < 2 || ny < 2 || nx * ny > 2 * idx.length || nx * ny > 4e6) return null;
    const ix = new Map(ux.map((v, k) => [v, k])), iy = new Map(uy.map((v, k) => [v, k]));
    const off = document.createElement("canvas");
    off.width = nx; off.height = ny;
    const octx = off.getContext("2d");
    const img = octx.createImageData(nx, ny);
    const cell = new Int32Array(nx * ny).fill(-1);
    for (const i of idx) {
      const cx = ix.get(X[i]), cy = ny - 1 - iy.get(Y[i]);
      const o = cy * nx + cx;
      const c = Math.max(0, Math.min(255, Math.round(((vals[i] - lo) / (hi - lo)) * 255)));
      img.data[o * 4] = lut[c * 3]; img.data[o * 4 + 1] = lut[c * 3 + 1];
      img.data[o * 4 + 2] = lut[c * 3 + 2]; img.data[o * 4 + 3] = 255;
      cell[o] = i;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, W, H);
    const pick = new Int32Array(W * H);
    for (let py = 0; py < H; py++) {
      const cy = Math.min(ny - 1, Math.floor((py / H) * ny));
      for (let px = 0; px < W; px++) pick[py * W + px] = cell[cy * nx + Math.min(nx - 1, Math.floor((px / W) * nx))];
    }
    return { pick, W, H };
  }

  // Scattered points: each claims its pixel, then grows into the empty pixels
  // around it for about one spacing between points. Gaps between samples
  // fill; a hole much larger than that — a body the flow goes round — stays
  // empty, as it should.
  function drawSplat(ctx, idx, X, Y, vals, lo, hi, lut, W, H) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const i of idx) {
      if (X[i] < x0) x0 = X[i]; if (X[i] > x1) x1 = X[i];
      if (Y[i] < y0) y0 = Y[i]; if (Y[i] > y1) y1 = Y[i];
    }
    let own = new Int32Array(W * H).fill(-1);
    for (const i of idx) {
      const px = Math.round(((X[i] - x0) / Math.max(1e-12, x1 - x0)) * (W - 1));
      const py = Math.round((1 - (Y[i] - y0) / Math.max(1e-12, y1 - y0)) * (H - 1));
      own[py * W + px] = i;
    }
    const reach = Math.max(1, Math.ceil(1.5 * Math.sqrt((W * H) / Math.max(1, idx.length))));
    for (let pass = 0; pass < reach; pass++) {
      const next = own.slice();
      let grew = false;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const o = y * W + x;
          if (own[o] >= 0) continue;
          const n = (x > 0 && own[o - 1] >= 0) ? own[o - 1] : (x < W - 1 && own[o + 1] >= 0) ? own[o + 1]
            : (y > 0 && own[o - W] >= 0) ? own[o - W] : (y < H - 1 && own[o + W] >= 0) ? own[o + W] : -1;
          if (n >= 0) { next[o] = n; grew = true; }
        }
      }
      own = next;
      if (!grew) break;
    }
    const img = ctx.createImageData(W, H);
    for (let o = 0; o < W * H; o++) {
      const i = own[o];
      if (i < 0) continue;
      const c = Math.max(0, Math.min(255, Math.round(((vals[i] - lo) / (hi - lo)) * 255)));
      img.data[o * 4] = lut[c * 3]; img.data[o * 4 + 1] = lut[c * 3 + 1];
      img.data[o * 4 + 2] = lut[c * 3 + 2]; img.data[o * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return { pick: own, W, H };
  }

  // a 1-D field: the profile as a line, its range in the labels underneath
  function drawLine(ctx, idx, X, vals, lo, hi, W, H, dpr) {
    let x0 = Infinity, x1 = -Infinity;
    for (const i of idx) { if (X[i] < x0) x0 = X[i]; if (X[i] > x1) x1 = X[i]; }
    const order = [...idx].sort((a, b) => X[a] - X[b]);
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, H - 1); ctx.lineTo(W, H - 1); ctx.stroke();
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2 * dpr; ctx.beginPath();
    order.forEach((i, k) => {
      const px = ((X[i] - x0) / Math.max(1e-12, x1 - x0)) * (W - 8 * dpr) + 4 * dpr;
      const py = (1 - (vals[i] - lo) / (hi - lo)) * (H - 8 * dpr) + 4 * dpr;
      if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    });
    ctx.stroke();
  }

  function drawBar(bar, lut) {
    const ctx = bar.getContext("2d");
    const img = ctx.createImageData(256, bar.height);
    for (let x = 0; x < 256; x++) for (let y = 0; y < bar.height; y++) {
      const o = (y * 256 + x) * 4;
      img.data[o] = lut[x * 3]; img.data[o + 1] = lut[x * 3 + 1]; img.data[o + 2] = lut[x * 3 + 2]; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  async function updatePlayer(run) {
    const host = document.getElementById("run-field");
    if (!S.player || S.player.runId !== run.id) S.player = new Player(host, run);
    await S.player.update(run.frames || []);
  }

  // ── activity feed ──
  const KIND = { tool: "$", text: "›", info: "•" };
  async function updateFeed(run) {
    const host = document.getElementById("run-feed");
    const block = S.selBlock;
    const key = `${run.id}:${block}`;
    if (!S.feed || S.feed.key !== key) {
      S.feed = { key, offset: null, n: 0 };
      host.replaceChildren();
    }
    const q = new URLSearchParams();
    if (block) q.set("block", block);
    if (S.feed.offset != null) q.set("after", String(S.feed.offset));
    const data = await get(`/api/runs/${encodeURIComponent(run.id)}/activity?${q}`);
    if (S.feed.key !== key) return;
    const atBottom = host.scrollTop + host.clientHeight >= host.scrollHeight - 30;
    const first = S.feed.offset == null;
    S.feed.offset = data.offset;
    document.getElementById("feed-aside").textContent = data.block ? `${data.block}${data.log ? ` · ${data.log}` : ""}` : "";
    if (first && !data.events.length) host.replaceChildren(h("div", { class: "empty", text: T("noActivity") }));
    if (data.events.length && host.querySelector(".empty")) host.replaceChildren();
    for (const ev of data.events) {
      const row = h("div", { class: `ev ${ev.kind}${S.feed.n++ % 2 ? " alt" : ""}${first ? "" : " fresh"}` },
        h("span", { class: "k", text: KIND[ev.kind] || "·" }), h("span", { class: "t", text: ev.text }));
      host.append(row);
      if (!first) setTimeout(() => row.classList.remove("fresh"), 50);
    }
    while (host.children.length > 400) host.firstChild.remove();
    if (first || atBottom) host.scrollTop = host.scrollHeight;
  }

  function renderEvalTable(host, run) {
    document.getElementById("evals-aside").textContent = run.evals.length ? String(run.evals.length) : "";
    const rows = run.evals.slice(-200).reverse().map((e) =>
      h("tr", { class: `clickable${e.block === S.selBlock ? " sel" : ""}`, title: e.error ? `✕ ${e.error}` : relTime(e.ts),
        onclick: () => { S.selBlock = e.block; refresh(); } },
        h("td", { text: e.block }), h("td", { class: "cand", text: e.candidate, title: e.candidate }),
        h("td", { class: "num", text: e.diag ? "diag" : fmtScore(e.rRMSE) }),
        h("td", { class: "num", text: e.wall_s != null ? String(Math.round(e.wall_s)) : "—" }),
        h("td", { text: e.error ? `✕ ${e.error.slice(0, 80)}` : relTime(e.ts) })));
    host.replaceChildren(h("table", {},
      h("thead", {}, h("tr", {}, h("th", { text: T("block") }), h("th", { text: T("candidate") }),
        h("th", { class: "num", text: T("score") }), h("th", { class: "num", text: T("wall") }), h("th", { text: T("when") }))),
      h("tbody", {}, rows)));
  }

  // ── markdown (a small, safe subset — text only ever via textContent) ──
  function inline(text) {
    const out = [];
    const re = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(text.slice(last, m.index));
      const tok = m[0];
      if (tok.startsWith("`")) out.push(h("code", { text: tok.slice(1, -1) }));
      else if (tok.startsWith("**")) out.push(h("b", { text: tok.slice(2, -2) }));
      else out.push(h("i", { text: tok.slice(1, -1) }));
      last = m.index + tok.length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }
  function markdown(src) {
    const root = h("div");
    const lines = String(src || "").split(/\r?\n/);
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.startsWith("```")) {
        const buf = [];
        i++;
        while (i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i++]);
        i++;
        root.append(h("pre", {}, h("code", { text: buf.join("\n") })));
        continue;
      }
      const hm = /^(#{1,3})\s+(.*)$/.exec(line);
      if (hm) { root.append(h(`h${hm[1].length}`, {}, inline(hm[2]))); i++; continue; }
      if (/^\s*\|/.test(line)) {
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
        const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const parsed = rows.map(cells).filter((r) => !r.every((c) => /^:?-+:?$/.test(c)));
        const [head, ...rest] = parsed;
        root.append(h("table", {}, h("thead", {}, h("tr", {}, (head || []).map((c) => h("th", {}, inline(c))))),
          h("tbody", {}, rest.map((r) => h("tr", {}, r.map((c) => h("td", {}, inline(c))))))));
        continue;
      }
      if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
        const ordered = /^\s*\d+\./.test(line);
        const list = h(ordered ? "ol" : "ul");
        const items = [];
        while (i < lines.length && lines[i].trim()) {
          const l = lines[i];
          if (/^\s*([-*]|\d+\.)\s+/.test(l)) items.push(l.replace(/^\s*([-*]|\d+\.)\s+/, ""));
          else if (/^(#|```|\s*\|)/.test(l)) break;
          else if (items.length) items[items.length - 1] += ` ${l.trim()}`;
          i++;
        }
        for (const it of items) list.append(h("li", {}, inline(it)));
        root.append(list);
        continue;
      }
      if (line.trim()) {
        const buf = [line];
        i++;
        while (i < lines.length && lines[i].trim() && !/^(#|```|\s*[-*|]\s|\s*\d+\.\s)/.test(lines[i])) buf.push(lines[i++]);
        root.append(h("p", {}, inline(buf.join(" "))));
        continue;
      }
      i++;
    }
    return root;
  }

  // ───────────────────────────── task view ─────────────────────────────
  async function viewTask(name, first) {
    const t = await get(`/api/tasks/${encodeURIComponent(name)}`);
    S.task = t;
    document.title = `${t.name} — PINNForge`;
    if (first || !app.querySelector("#task-steps")) {
      app.replaceChildren(
        h("div", { class: "page-title" }, h("h1", { text: t.name }), h("span", { class: "meta", id: "task-title" }),
          h("span", { id: "task-state" })),
        h("div", { id: "task-ready" }),
        h("div", { class: "grid-2" },
          h("section", { class: "card" }, h("h2", {}, icon("authoring"), T("stages"), h("span", { class: "aside", id: "task-mode" })),
            h("div", { id: "task-steps", class: "stepper" })),
          h("section", { class: "card" }, h("h2", {}, icon("reference"), T("reference")), h("div", { id: "task-ref" }))),
        h("div", { class: "grid-2 even" },
          h("section", { class: "card scroll" }, h("h2", {}, icon("agent"), T("authorAgent")), h("div", { id: "task-feed", class: "feed" })),
          h("section", { class: "card scroll" }, h("h2", {}, icon("problem"), T("problem")), h("div", { id: "task-problem", class: "md" }))),
      );
      S.feed = { key: `task:${name}`, offset: null, n: 0 };
    }
    document.getElementById("task-title").textContent = t.title || "";
    document.getElementById("task-state").replaceChildren(statusEl(t.ready ? "ready" : ((t.authoring || {}).status === "failed" ? "failed" : "running")));
    const ready = document.getElementById("task-ready");
    ready.replaceChildren(t.ready ? h("div", { class: "card" }, h("span", { text: T("runIt") }),
      h("div", { class: "hint", text: `pinnforge run start -t ${t.name} -n 3` })) : "");
    renderSteps(document.getElementById("task-steps"), t);
    renderReference(document.getElementById("task-ref"), t);
    const prob = document.getElementById("task-problem");
    prob.replaceChildren(t.problem ? markdown(t.problem) : h("div", { class: "empty", text: "—" }));
    await updateTaskFeed(name);
  }

  function renderSteps(host, t) {
    const a = t.authoring;
    document.getElementById("task-mode").textContent = a ? (a.mode === "ingest" ? T("refOperator") : T("refBuilt")) : "";
    if (!a) { host.replaceChildren(statusEl("ready")); return; }
    host.replaceChildren(...a.plan.map((stage, i) => {
      const st = (a.stages || {})[stage] || {};
      const status = st.status || "pending";
      const mark = { done: "✓", failed: "✕", skipped: "–" }[status] || String(i + 1);
      const took = st.started && st.ended ? fmtDur((Date.parse(st.ended) - Date.parse(st.started)) / 1000)
        : st.started ? relTime(st.started) : "";
      return h("div", { class: `step ${status}` }, h("div", { class: "dot", text: mark }),
        h("div", { class: "what" }, h("b", { text: T("stage_" + stage) }), st.note ? h("div", { class: "note", text: st.note }) : null),
        h("div", { class: "when", text: took }));
    }));
  }

  function renderReference(host, t) {
    const r = t.reference;
    if (!r) { host.replaceChildren(h("div", { class: "empty", text: "—" })); return; }
    const rows = [];
    const row = (k, v) => rows.push(h("tr", {}, h("td", { text: k }), h("td", { class: "num" }, v)));
    row(T("status"), statusEl(r.verdict === "pass" ? "done" : "failed"));
    row(T("reference"), r.source === "operator" ? T("refOperator") : T("refBuilt"));
    if (r.estimate != null) row(T("estimate"), fmtScore(r.estimate));
    if (r.order != null) row(T("order"), r.order.toFixed(2));
    if (r.reproducible != null) row(T("reproduced"), r.reproducible ? T("yes") : T("no"));
    const parts = [h("table", {}, h("tbody", {}, rows))];
    if (r.differences && r.differences.length) parts.push(convergenceChart(r, host.clientWidth));
    for (const reason of r.reasons || []) parts.push(h("div", { class: "note", text: `✕ ${reason}` }));
    host.replaceChildren(...parts);
  }

  function convergenceChart(r, width) {
    // drawn at the width it is shown at, so its text stays the size of the page's
    const W = Math.max(320, Math.min(900, width || 420)), H = 220, m = { l: 52, r: 14, t: 14, b: 30 };
    const d = r.differences;
    const all = d.concat(r.estimate ? [r.estimate] : [], r.tol ? [r.tol] : []).filter((v) => v > 0);
    let lo = Math.floor(Math.log10(Math.min(...all))), hi = Math.ceil(Math.log10(Math.max(...all)));
    if (hi === lo) hi += 1;
    const n = d.length + (r.estimate ? 1 : 0);
    const x = (i) => m.l + (n <= 1 ? 0.5 : i / (n - 1)) * (W - m.l - m.r);
    const y = (v) => m.t + ((hi - Math.log10(v)) / (hi - lo)) * (H - m.t - m.b);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": T("levelDiff") });
    for (let k = lo; k <= hi; k++) {
      svg.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y(10 ** k), y2: y(10 ** k) }),
        s("text", { class: "tick", x: m.l - 6, y: y(10 ** k) + 4, "text-anchor": "end", text: `1e${k}` }));
    }
    if (r.tol) {
      svg.append(s("line", { class: "tol-line", x1: m.l, x2: W - m.r, y1: y(r.tol), y2: y(r.tol) }),
        s("text", { class: "tol-label", x: W - m.r, y: y(r.tol) - 5, "text-anchor": "end", text: `${T("tolerance")} ${fmtScore(r.tol)}` }));
    }
    const line = d.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("");
    svg.append(s("path", { class: "conv-line", d: line }));
    d.forEach((v, i) => {
      svg.append(s("circle", { class: "conv-pt", cx: x(i), cy: y(v), r: 4.5 }),
        s("text", { class: "tick", x: x(i), y: H - 10, "text-anchor": "middle", text: `L${i + 1}→L${i + 2}` }),
        s("circle", { class: "hit", cx: x(i), cy: y(v), r: 12,
          onpointermove: (e) => showTip(e, [h("div", { class: "v", text: fmtScore(v) }), h("div", { text: `L${i + 1} vs L${i + 2}` })]),
          onpointerleave: hideTip }));
    });
    if (r.estimate) {
      const i = d.length;
      svg.append(s("line", { x1: x(i - 1), y1: y(d[d.length - 1]), x2: x(i), y2: y(r.estimate), stroke: css("--accent"), "stroke-width": 1.5, "stroke-dasharray": "3 3" }),
        s("circle", { cx: x(i), cy: y(r.estimate), r: 5, fill: css("--surface"), stroke: css("--accent"), "stroke-width": 2 }),
        s("text", { class: "tick", x: x(i), y: H - 10, "text-anchor": "middle", text: "est." }));
    }
    return h("div", {}, svg, h("div", { class: "legend" },
      h("span", {}, h("i", { class: "k-line" }), T("levelDiff")),
      h("span", {}, h("i", { class: "k-hollow" }), T("finestEst"))));
  }

  async function updateTaskFeed(name) {
    const host = document.getElementById("task-feed");
    const q = S.feed && S.feed.offset != null ? `?after=${S.feed.offset}` : "";
    let data;
    try { data = await get(`/api/tasks/${encodeURIComponent(name)}/activity${q}`); } catch { return; }
    const first = S.feed.offset == null;
    S.feed.offset = data.offset;
    if (first && !data.events.length) host.replaceChildren(h("div", { class: "empty", text: T("noActivity") }));
    if (data.events.length && host.querySelector(".empty")) host.replaceChildren();
    for (const ev of data.events) {
      host.append(h("div", { class: `ev ${ev.kind}${S.feed.n++ % 2 ? " alt" : ""}` }, h("span", { class: "k", text: KIND[ev.kind] || "·" }),
        h("span", { class: "t", text: ev.text })));
    }
    host.scrollTop = host.scrollHeight;
  }

  // ───────────────────────────── chrome ─────────────────────────────
  function applyTheme(theme) {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
  }
  applyTheme(store.get("pf-theme"));
  document.getElementById("theme-btn").addEventListener("click", () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark"
      || (!document.documentElement.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    store.set("pf-theme", next);
    applyTheme(next);
    S.frames.clear();
    route();
  });
  window.addEventListener("hashchange", route);
  window.addEventListener("resize", () => {
    clearTimeout(window.__pfr);
    window.__pfr = setTimeout(() => { fitScreen(); refresh(); }, 200);
  });
  // a stream that dies silently still gets a refresh every few seconds
  if (!window.__PF_STATIC__) setInterval(() => { if (!document.hidden) refresh(); }, 15000);

  // what replay.js draws with: the API and this page's own field renderers
  window.PF = { get, field: { resolveView, panelValues, drawTensor, drawSplat, drawLine } };

  get("/api/ping").then((p) => {
    S.ws = { name: p.workspace || "", path: p.path || "" };
    if (S.run && document.getElementById("run-meta")) updateRunHeader(S.run);
  }).catch(() => {});
  connect();
  route();
})();
