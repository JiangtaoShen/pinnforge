/* PINNForge dashboard — vanilla JS, no build step, no third-party code.
 *
 * Everything agents wrote (summaries, transcripts, file names) is inserted
 * with textContent: the page never parses agent text as HTML.
 */
"use strict";
(() => {
  // ───────────────────────────── i18n ─────────────────────────────
  const I18N = {
    en: {
      runs: "Runs", tasks: "Tasks", compute: "Compute", noRuns: "No runs yet.",
      startHint: "Start one:", noTasks: "No tasks yet.", live: "live", idle: "idle",
      connecting: "connecting…", connected: "live", best: "Best rRMSE", anchor: "Anchor (b00)",
      vsAnchor: "{0}× better than the anchor", evals: "Evaluations", errors: "{0} failed",
      gpuTime: "GPU time", cpuTime: "CPU time", cost: "Model cost", tokens: "{0} tokens", progress: "Progress",
      progressSub: "Every scored evaluation, by cumulative {0}; the line is the best so far.",
      legendEval: "evaluation", legendCur: "current block", legendBest: "best so far",
      legendAnchor: "anchor", legendErr: "failed evaluation", field: "Predicted field",
      fieldSub: "The best predictions so far, as the blocks found them.",
      noFrames: "The field appears here once an evaluation is scored.",
      play: "Play", pause: "Pause", latest: "Latest", timeSlice: "t =", agent: "Agent activity",
      noActivity: "No agent activity yet.", blocks: "Blocks", summary: "Summary",
      noSummary: "This block has not written its summary yet.", evalTable: "Evaluations",
      block: "Block", candidate: "Candidate", score: "rRMSE", wall: "Wall s", when: "When",
      status: "Status", spentOf: "{0} / {1} s", nEvals: "{0} evals", gpus: "GPUs", util: "util",
      mem: "mem", usedBy: "in use by {0}", free: "free", notInPool: "not used by this workspace", noGpu: "No GPU visible — CPU mode.",
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
    },
    zh: {
      runs: "运行", tasks: "任务", compute: "算力", noRuns: "还没有运行。",
      startHint: "开始一个：", noTasks: "还没有任务。", live: "运行中", idle: "空闲",
      connecting: "连接中…", connected: "实时", best: "最佳 rRMSE", anchor: "基线 (b00)",
      vsAnchor: "比基线好 {0} 倍", evals: "评估次数", errors: "{0} 次失败",
      gpuTime: "GPU 时间", cpuTime: "CPU 时间", cost: "模型费用", tokens: "{0} tokens", progress: "进展",
      progressSub: "每次已评分的评估，按累计 {0}排列；折线是迄今最佳。",
      legendEval: "评估", legendCur: "当前 block", legendBest: "迄今最佳",
      legendAnchor: "基线", legendErr: "失败的评估", field: "预测场",
      fieldSub: "各 block 找到的迄今最佳预测，按时间回放。",
      noFrames: "有评估完成评分后，这里会显示预测场。",
      play: "播放", pause: "暂停", latest: "最新", timeSlice: "t =", agent: "智能体动态",
      noActivity: "还没有智能体动态。", blocks: "Blocks", summary: "总结",
      noSummary: "这个 block 还没有写总结。", evalTable: "评估记录",
      block: "Block", candidate: "候选", score: "rRMSE", wall: "耗时 s", when: "时间",
      status: "状态", spentOf: "{0} / {1} 秒", nEvals: "{0} 次评估", gpus: "GPU", util: "利用率",
      mem: "显存", usedBy: "被 {0} 使用", free: "空闲", notInPool: "本工作区不使用", noGpu: "没有可见的 GPU —— CPU 模式。",
      model: "模型", task: "任务", title: "标题", state: "状态", ready: "就绪",
      authoring: "创建中：{0}", failedAt: "失败于 {0}", stages: "任务创建",
      reference: "参考解", refBuilt: "由智能体构建", refOperator: "由用户提供",
      estimate: "估计误差", order: "观测收敛阶",
      reproduced: "可复现", yes: "是", no: "否", tolerance: "容差",
      levelDiff: "相邻两层之差", finestEst: "最细层的误差估计",
      authorAgent: "任务编写智能体", problem: "problem.md", runIt: "运行它：",
      notFound: "未找到。", workspace: "工作区", back: "总览", tFieldAt: "于",
      stage_intake: "收集需求", stage_author: "编写任务包", stage_reference: "参考解",
      stage_reference_check: "检查参考解", stage_validate: "契约校验",
      stage_smoke: "冒烟测试", stage_anchor: "测量基线", stage_finalize: "就绪",
      st_done: "完成", st_running: "进行中", st_failed: "失败", st_short: "未用满",
      st_interrupted: "已中断", st_stopped: "已停止", st_finished: "已结束",
      st_created: "已创建", st_pending: "等待", st_skipped: "跳过", st_ready: "就绪",
      st_measuring: "测量中", st_missing: "缺失", st_unreadable: "无法读取",
      st_clean: "完整", st_violated: "被改动", st_unchecked: "未检查",
      integrity: "完整性", blocksOf: "{0}/{1} 个 block", table: "表格", snapshot: "快照",
    },
  };
  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  let lang = store.get("pf-lang") || ((navigator.language || "").startsWith("zh") ? "zh" : "en");
  const T = (key, ...args) => {
    let text = (I18N[lang] && I18N[lang][key]) ?? I18N.en[key] ?? key;
    args.forEach((a, i) => { text = text.replace(`{${i}}`, a); });
    return text;
  };
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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
    const fmt = new Intl.RelativeTimeFormat(lang === "zh" ? "zh-CN" : "en", { numeric: "auto" });
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
  function showTip(evt, rows) {
    tip.replaceChildren(...rows);
    tip.hidden = false;
    const pad = 14, w = tip.offsetWidth, hgt = tip.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + hgt > window.innerHeight - 8) y = evt.clientY - hgt - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const hideTip = () => { tip.hidden = true; };

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
    run: null, runSeen: {}, player: null, feed: null, selBlock: null, task: null,
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
    S.feed = null;
    S.selBlock = null;
    app.replaceChildren();
    crumbs();
    await refresh(true);
  }
  function crumbs() {
    const c = document.getElementById("crumbs");
    const [kind, id] = S.route;
    const items = [h("a", { href: "#/", text: T("back") })];
    if (kind === "run" && id) items.push(h("span", { class: "sep", text: "/" }), h("span", { text: id }));
    if (kind === "task" && id) items.push(h("span", { class: "sep", text: "/" }), h("span", { text: id }));
    c.replaceChildren(...items);
  }

  let refreshing = false, again = false;
  async function refresh(first = false) {
    if (refreshing) { again = true; return; }
    refreshing = true;
    try {
      const [kind, id] = S.route;
      if (kind === "run" && id) await viewRun(id, first);
      else if (kind === "task" && id) await viewTask(id, first);
      else await viewOverview(first);
    } catch (e) {
      if (e.status === 404) app.replaceChildren(h("div", { class: "card empty", text: T("notFound") }));
      else console.error(e);
    } finally {
      refreshing = false;
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
    document.getElementById("ws-name").textContent = data.name;
    document.title = `PINNForge — ${data.name}`;
    if (first || !app.querySelector("#ov-runs")) {
      app.replaceChildren(
        h("div", { class: "page-title" }, h("h1", { text: data.name }),
          h("span", { class: "meta", text: `${data.workspace} · pinnforge ${data.version}` })),
        h("section", { class: "card" }, h("h2", {}, T("compute")), h("div", { id: "ov-gpus", class: "gpus" })),
        h("section", {}, h("h2", { class: "section-h", text: T("runs") }), h("div", { id: "ov-runs", class: "grid-3" })),
        h("section", { class: "card" }, h("h2", {}, T("tasks")), h("div", { id: "ov-tasks" })),
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
      const temp = g.temp != null ? `${Math.round(g.temp)} °C` : "";
      const power = g.power != null ? ` · ${Math.round(g.power)} W` : "";
      const unused = Array.isArray(pool) && !pool.includes(g.index);
      card.classList.toggle("unused", unused);
      const who = card.querySelector(".who");
      who.replaceChildren(holder
        ? h("a", { href: `#/run/${encodeURIComponent(holder)}`, text: T("usedBy", holder) })
        : h("span", { text: T(unused ? "notInPool" : "free") }), ` · ${temp}${power}`);
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
    renderScoreChart(document.getElementById("run-chart"), run, prev);
    renderGantt(document.getElementById("run-gantt"), run);
    await Promise.all([updatePlayer(run), updateFeed(run), updateSummary(run)]);
    renderEvalTable(document.getElementById("run-table"), run);
  }

  // a run with no card trains on the CPU; its clock is not a GPU's
  const timeLabel = (run) => T(((run.config && run.config.gpus) || []).length ? "gpuTime" : "cpuTime");

  function buildRunLayout(run) {
    app.replaceChildren(
      h("div", { class: "page-title" }, h("h1", { id: "run-title" }), h("span", { class: "meta", id: "run-meta" }),
        h("span", { id: "run-status" })),
      h("div", { class: "tiles", id: "run-tiles" }),
      h("section", { class: "card" }, h("h2", {}, T("progress")), h("p", { class: "sub", text: T("progressSub", timeLabel(run)) }),
        h("div", { id: "run-chart" }),
        h("div", { class: "legend" },
          h("span", {}, h("i", { class: "k-dot" }), T("legendEval")),
          h("span", {}, h("i", { class: "k-cur" }), T("legendCur")),
          h("span", {}, h("i", { class: "k-line" }), T("legendBest")),
          h("span", {}, h("i", { class: "k-dash" }), T("legendAnchor")),
          h("span", {}, h("i", { class: "k-err" }), T("legendErr")))),
      h("div", { class: "grid-2" },
        h("section", { class: "card" }, h("h2", {}, T("field"), h("span", { class: "aside", id: "field-aside" })),
          h("p", { class: "sub", text: T("fieldSub") }), h("div", { id: "run-field", class: "field" })),
        h("section", { class: "card" }, h("h2", {}, T("agent"), h("span", { class: "aside", id: "feed-aside" })),
          h("div", { id: "run-feed", class: "feed", "aria-live": "polite" }))),
      h("section", { class: "card" }, h("h2", {}, T("blocks")), h("div", { id: "run-gantt", class: "gantt" })),
      h("section", { class: "card" }, h("h2", { id: "sum-h" }, T("summary")), h("div", { id: "run-summary", class: "md" })),
      h("section", { class: "card" }, h("h2", {}, T("evalTable")), h("div", { id: "run-table" })),
    );
  }

  function updateRunHeader(run) {
    document.getElementById("run-title").textContent = run.id;
    const agent = run.config && run.config.agent ? `${run.config.agent.runtime} / ${run.config.agent.model || "default"}` : "";
    const gp = (run.config && run.config.gpus) || [];
    const hw = run.config && run.config.hw && run.config.hw.gpu ? run.config.hw.gpu : "";
    document.getElementById("run-meta").textContent =
      `${run.task} · ${agent} · ${gp.length ? `GPU ${gp.join(", ")}${hw ? ` (${hw})` : ""}` : "CPU"}`;
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
        tile("t-gpu", timeLabel(run)), tile("t-cost", T("cost")));
      host.firstChild.classList.add("hero");
    }
    const q = (id, sel) => document.querySelector(`#${id} ${sel}`);
    tween(q("t-best", ".value"), run.best, fmtScore);
    const ratio = run.best && run.anchor ? run.anchor / run.best : null;
    q("t-best", ".note").replaceChildren(ratio && ratio > 1.0001
      ? h("span", { class: "delta-good", text: `▲ ${T("vsAnchor", ratio.toFixed(ratio >= 10 ? 0 : 1))}` })
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
    const spent = run.blocks.filter((b) => b.id !== "b00").reduce((a, b) => a + (b.spent || 0), 0);
    const budget = (run.config.wall_budget_s || 0) * (run.blocks_total || 0);
    tween(q("t-gpu", ".value"), spent, fmtDur);
    q("t-gpu", ".note").textContent = budget ? `/ ${fmtDur(budget)}` : "";
    const cost = run.blocks.reduce((a, b) => a + (b.cost_usd || 0), 0);
    const tokens = run.blocks.reduce((a, b) => a + (b.tokens || 0), 0);
    tween(q("t-cost", ".value"), cost, (v) => (cost ? `$${v.toFixed(2)}` : "—"));
    q("t-cost", ".note").textContent = tokens ? T("tokens", fmtCompact(tokens)) : "";
  }

  // ── score chart ──
  function renderScoreChart(host, run, prev) {
    const W = Math.max(320, host.clientWidth || 800), H = 300;
    const m = { l: 56, r: 16, t: 22, b: 34 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
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
    const x = (v) => m.l + (v / maxX) * iw;
    const y = (v) => m.t + ((hi - Math.log10(v)) / (hi - lo)) * ih;

    // block bands
    const blockSpans = [];
    for (const e of evs) {
      const last = blockSpans[blockSpans.length - 1];
      if (!last || last.block !== e.block) blockSpans.push({ block: e.block, x0: (e.cum_wall || 0) - (e.wall_s || 0), x1: e.cum_wall || 0 });
      else last.x1 = e.cum_wall || 0;
    }
    blockSpans.forEach((b, i) => {
      if (i % 2 === 1) svg.append(s("rect", { class: "band", x: x(Math.max(0, b.x0)), y: m.t, width: Math.max(0, x(b.x1) - x(Math.max(0, b.x0))), height: ih }));
      const cx = (x(Math.max(0, b.x0)) + x(b.x1)) / 2;
      if (x(b.x1) - x(Math.max(0, b.x0)) > 26) svg.append(s("text", { class: "band-label", x: cx, y: m.t - 8, "text-anchor": "middle", text: b.block }));
    });
    // grid + ticks (decades and half-decades)
    for (let d = Math.ceil(lo * 2) / 2; d <= hi + 1e-9; d += 0.5) {
      const v = Math.pow(10, d), yy = y(v);
      svg.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: yy, y2: yy }),
        s("text", { class: "tick", x: m.l - 8, y: yy + 4, "text-anchor": "end", text: fmtScore(Number(v.toPrecision(2))) }));
    }
    svg.append(s("line", { class: "axis", x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }));
    const step = niceStep(maxX / 6);
    for (let v = 0; v <= maxX + 1e-9; v += step) {
      svg.append(s("text", { class: "tick", x: x(v), y: H - m.b + 16, "text-anchor": "middle", text: fmtDur(v) }));
    }
    svg.append(s("text", { class: "axis-label", x: W - m.r, y: H - 4, "text-anchor": "end", text: `${timeLabel(run)} →` }));
    svg.append(s("text", { class: "axis-label", x: 4, y: 12, text: "rRMSE (log)" }));
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
  function colormap(diverging) {
    const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
    const mid = css("--surface-2").startsWith("#") ? css("--surface-2") : "#f0efec";
    const stops = diverging
      ? ["#104281", "#3987e5", "#9ec5f4", mid, "#f2a5a0", "#e34948", "#9d2b2a"].map(hex)
      : ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"].map(hex);
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const f = (i / 255) * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(f)), u = f - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * u;
    }
    return lut;
  }

  class Player {
    constructor(host, run) {
      this.host = host;
      this.runId = run.id;
      this.dims = (run.grid && run.grid.dims) || 2;
      this.names = (run.grid && run.grid.coords) || ["x", "y"];
      this.frames = [];
      this.index = -1;
      this.field = null;
      this.tIndex = null;
      this.follow = true;
      this.playing = false;
      this.canvas = h("canvas", { width: 640, height: 400, "aria-label": T("field"),
        onpointermove: (e) => this.hover(e), onpointerleave: hideTip });
      this.caption = h("div", { class: "caption" });
      this.bar = h("canvas", { width: 256, height: 10 });
      this.lo = h("span"); this.hi = h("span");
      this.slider = h("input", { type: "range", min: "0", max: "0", value: "0",
        oninput: () => { this.follow = false; this.show(Number(this.slider.value)); } });
      this.playBtn = h("button", { class: "btn", type: "button", onclick: () => this.toggle() }, T("play"));
      this.latestBtn = h("button", { class: "btn", type: "button", onclick: () => { this.follow = true; this.show(this.frames.length - 1); } }, T("latest"));
      this.fieldSel = h("select", { onchange: () => { this.field = this.fieldSel.value; this.draw(); } });
      this.tSel = h("input", { type: "range", min: "0", max: "0", value: "0", hidden: true,
        oninput: () => { this.tIndex = Number(this.tSel.value); this.draw(); } });
      this.tLabel = h("span", { class: "muted" });
      host.replaceChildren(this.caption, this.canvas,
        h("div", { class: "colorbar" }, this.lo, this.bar, this.hi),
        h("div", { class: "controls" }, this.playBtn, this.slider, this.latestBtn, this.fieldSel),
        h("div", { class: "controls" }, this.tLabel, this.tSel));
    }
    stop() { clearInterval(this.timer); this.playing = false; }
    toggle() {
      if (this.playing) { this.stop(); this.playBtn.textContent = T("play"); return; }
      this.playing = true;
      this.follow = false;
      this.playBtn.textContent = T("pause");
      let i = 0;
      this.show(0);
      this.timer = setInterval(() => {
        i += 1;
        if (i >= this.frames.length) { this.stop(); this.playBtn.textContent = T("play"); this.follow = true; return; }
        this.show(i);
      }, reduced ? 1400 : 900);
    }
    async update(frames) {
      const fresh = frames.filter((f) => f.run_best);
      const changed = fresh.length !== this.frames.length;
      this.frames = fresh;
      this.slider.max = String(Math.max(0, fresh.length - 1));
      document.getElementById("field-aside").textContent = fresh.length ? `${fresh.length}` : "";
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
      const names = Object.keys(this.data.fields);
      if (!this.field || !names.includes(this.field)) this.field = names[0];
      this.fieldSel.replaceChildren(...names.map((n) => h("option", { value: n, text: n, selected: n === this.field })));
      this.fieldSel.hidden = names.length < 2;
      this.caption.replaceChildren(h("b", { text: fmtScore(meta.rRMSE) }),
        h("span", { text: `${meta.block} · ${String(meta.candidate).split("/").pop()}` }),
        h("span", { class: "muted", text: `${i + 1} / ${this.frames.length}` }));
      if (!reduced) {
        this.canvas.style.opacity = "0.35";
        requestAnimationFrame(() => { this.draw(); this.canvas.style.opacity = "1"; });
      } else this.draw();
    }
    draw() {
      const d = this.data;
      if (!d || !this.field) return;
      const vals = d.fields[this.field].values, coords = d.coords;
      let X = coords[0], Y = coords[1] || null, keep = null;
      if (this.dims >= 3 && coords.length >= 3) {
        const ts = [...new Set(coords[0])].sort((a, b) => a - b);
        if (this.tIndex == null || this.tIndex >= ts.length) this.tIndex = ts.length - 1;
        this.tSel.hidden = false;
        this.tSel.max = String(ts.length - 1);
        this.tSel.value = String(this.tIndex);
        const t = ts[this.tIndex];
        this.tLabel.textContent = `${this.names[0] || "t"} = ${Number(t.toPrecision(4))}`;
        keep = coords[0].map((v, i) => (v === t ? i : -1)).filter((i) => i >= 0);
        X = coords[1]; Y = coords[2];
      }
      const idx = keep || vals.map((_, i) => i);
      let lo = Infinity, hi = -Infinity;
      for (const i of idx) { const v = vals[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      // diverging (centred on zero) only when both signs carry real weight;
      // a mostly-positive field with small negative ripples reads better on
      // one hue from its minimum to its maximum
      const diverging = lo < 0 && hi > 0 && Math.min(-lo, hi) >= 0.2 * Math.max(-lo, hi);
      if (diverging) { const a = Math.max(-lo, hi); lo = -a; hi = a; }
      if (hi === lo) hi = lo + 1;
      const lut = colormap(diverging);
      const cw = this.host.clientWidth || 600;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const i of idx) {
        x0 = Math.min(x0, X[i]); x1 = Math.max(x1, X[i]);
        if (Y) { y0 = Math.min(y0, Y[i]); y1 = Math.max(y1, Y[i]); }
      }
      const W = Math.round(cw * dpr);
      if (!Y) { this.draw1d(idx, X, vals, lo, hi, x0, x1, W, dpr); return; }
      const aspect = Math.min(1.1, Math.max(0.35, (y1 - y0) / Math.max(1e-12, x1 - x0)));
      const Hh = Math.round(W * aspect);
      this.canvas.width = W; this.canvas.height = Hh;
      this.canvas.style.height = `${Math.round(Hh / dpr)}px`;
      const ctx = this.canvas.getContext("2d");
      if (this.drawTensor(ctx, idx, X, Y, vals, lo, hi, lut, W, Hh, dpr)) return;
      const img = ctx.createImageData(W, Hh);
      const pick = new Int32Array(W * Hh).fill(-1);
      const n = idx.length;
      const size = Math.max(1, Math.ceil(Math.sqrt((W * Hh) / n) * 1.25));
      const half = Math.floor(size / 2);
      for (const i of idx) {
        const px = Math.round(((X[i] - x0) / Math.max(1e-12, x1 - x0)) * (W - 1));
        const py = Math.round((1 - (Y[i] - y0) / Math.max(1e-12, y1 - y0)) * (Hh - 1));
        const c = Math.max(0, Math.min(255, Math.round(((vals[i] - lo) / (hi - lo)) * 255)));
        for (let dy = -half; dy < size - half; dy++) {
          const yy = py + dy;
          if (yy < 0 || yy >= Hh) continue;
          for (let dx = -half; dx < size - half; dx++) {
            const xx = px + dx;
            if (xx < 0 || xx >= W) continue;
            const o = yy * W + xx;
            img.data[o * 4] = lut[c * 3]; img.data[o * 4 + 1] = lut[c * 3 + 1];
            img.data[o * 4 + 2] = lut[c * 3 + 2]; img.data[o * 4 + 3] = 255;
            pick[o] = i;
          }
        }
      }
      ctx.putImageData(img, 0, 0);
      this.pick = { pick, W, Hh, dpr, X, Y, vals };
      this.drawBar(lut, lo, hi);
    }
    drawTensor(ctx, idx, X, Y, vals, lo, hi, lut, W, Hh, dpr) {
      // A tensor-product grid renders as an image of its own cells, scaled
      // up with smoothing: exact, and far crisper than point splats.
      const ux = [...new Set(idx.map((i) => X[i]))].sort((a, b) => a - b);
      const uy = [...new Set(idx.map((i) => Y[i]))].sort((a, b) => a - b);
      const nx = ux.length, ny = uy.length;
      if (nx < 2 || ny < 2 || nx * ny !== idx.length || nx * ny > 4e6) return false;
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
      ctx.drawImage(off, 0, 0, W, Hh);
      // hover: display pixel -> grid cell -> point
      const pick = new Int32Array(W * Hh);
      for (let py = 0; py < Hh; py++) {
        const cy = Math.min(ny - 1, Math.floor((py / Hh) * ny));
        for (let px = 0; px < W; px++) pick[py * W + px] = cell[cy * nx + Math.min(nx - 1, Math.floor((px / W) * nx))];
      }
      this.pick = { pick, W, Hh, dpr, X, Y, vals };
      this.drawBar(lut, lo, hi);
      return true;
    }
    draw1d(idx, X, vals, lo, hi, x0, x1, W, dpr) {
      const Hh = Math.round(W * 0.4);
      this.canvas.width = W; this.canvas.height = Hh;
      this.canvas.style.height = `${Math.round(Hh / dpr)}px`;
      const ctx = this.canvas.getContext("2d");
      ctx.clearRect(0, 0, W, Hh);
      const order = [...idx].sort((a, b) => X[a] - X[b]);
      ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2 * dpr; ctx.beginPath();
      order.forEach((i, k) => {
        const px = ((X[i] - x0) / Math.max(1e-12, x1 - x0)) * (W - 8 * dpr) + 4 * dpr;
        const py = (1 - (vals[i] - lo) / (hi - lo)) * (Hh - 8 * dpr) + 4 * dpr;
        if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      });
      ctx.stroke();
      this.pick = null;
      this.drawBar(colormap(false), lo, hi);
    }
    drawBar(lut, lo, hi) {
      const ctx = this.bar.getContext("2d");
      const img = ctx.createImageData(256, 10);
      for (let x = 0; x < 256; x++) for (let y = 0; y < 10; y++) {
        const o = (y * 256 + x) * 4;
        img.data[o] = lut[x * 3]; img.data[o + 1] = lut[x * 3 + 1]; img.data[o + 2] = lut[x * 3 + 2]; img.data[o + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      this.lo.textContent = fmtScore(lo); this.hi.textContent = fmtScore(hi);
    }
    hover(evt) {
      if (!this.pick) return;
      const r = this.canvas.getBoundingClientRect();
      const px = Math.floor(((evt.clientX - r.left) / r.width) * this.pick.W);
      const py = Math.floor(((evt.clientY - r.top) / r.height) * this.pick.Hh);
      const i = this.pick.pick[py * this.pick.W + px];
      if (i == null || i < 0) { hideTip(); return; }
      const nx = this.dims >= 3 ? this.names[1] : this.names[0];
      const ny = this.dims >= 3 ? this.names[2] : this.names[1];
      showTip(evt, [h("div", { class: "v", text: `${this.field} = ${fmtScore(this.pick.vals[i])}` }),
        h("div", { text: `${nx || "x"} ${Number(this.pick.X[i].toPrecision(4))} · ${ny || "y"} ${Number(this.pick.Y[i].toPrecision(4))}` })]);
    }
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
      S.feed = { key, offset: null };
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
      const row = h("div", { class: `ev ${ev.kind}${first ? "" : " fresh"}` },
        h("span", { class: "k", text: KIND[ev.kind] || "·" }), h("span", { class: "t", text: ev.text }));
      host.append(row);
      if (!first) setTimeout(() => row.classList.remove("fresh"), 50);
    }
    while (host.children.length > 400) host.firstChild.remove();
    if (first || atBottom) host.scrollTop = host.scrollHeight;
  }

  // ── summary ──
  async function updateSummary(run) {
    const host = document.getElementById("run-summary");
    const blk = run.blocks.find((b) => b.id === S.selBlock);
    document.getElementById("sum-h").textContent = `${T("summary")}${S.selBlock ? ` · ${S.selBlock}` : ""}`;
    if (!blk || !blk.summary) { host.replaceChildren(h("div", { class: "empty", text: T("noSummary") })); return; }
    if (host.dataset.key === `${run.id}:${blk.id}:${run.updated}`) return;
    const data = await get(`/api/runs/${encodeURIComponent(run.id)}/summary/${blk.id}`);
    host.dataset.key = `${run.id}:${blk.id}:${run.updated}`;
    host.replaceChildren(markdown(data.markdown));
  }

  function renderEvalTable(host, run) {
    const rows = run.evals.slice(-40).reverse().map((e) =>
      h("tr", {}, h("td", { text: e.block }), h("td", { text: e.candidate }),
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
    const lines = String(src || "").split("\n");
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
          h("section", { class: "card" }, h("h2", {}, T("stages"), h("span", { class: "aside", id: "task-mode" })),
            h("div", { id: "task-steps", class: "stepper" })),
          h("section", { class: "card" }, h("h2", {}, T("reference")), h("div", { id: "task-ref" }))),
        h("section", { class: "card" }, h("h2", {}, T("authorAgent")), h("div", { id: "task-feed", class: "feed" })),
        h("section", { class: "card" }, h("h2", {}, T("problem")), h("div", { id: "task-problem", class: "md" })),
      );
      S.feed = { key: `task:${name}`, offset: null };
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
    if (r.differences && r.differences.length) parts.push(convergenceChart(r));
    for (const reason of r.reasons || []) parts.push(h("div", { class: "note", text: `✕ ${reason}` }));
    host.replaceChildren(...parts);
  }

  function convergenceChart(r) {
    const W = 420, H = 190, m = { l: 52, r: 14, t: 14, b: 30 };
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
      host.append(h("div", { class: `ev ${ev.kind}` }, h("span", { class: "k", text: KIND[ev.kind] || "·" }),
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
  const langBtn = document.getElementById("lang-btn");
  const setLangLabel = () => { langBtn.textContent = lang === "zh" ? "EN" : "中"; document.documentElement.lang = lang === "zh" ? "zh-CN" : "en"; };
  setLangLabel();
  langBtn.addEventListener("click", () => {
    lang = lang === "zh" ? "en" : "zh";
    store.set("pf-lang", lang);
    setLangLabel();
    route();
  });
  window.addEventListener("hashchange", route);
  window.addEventListener("resize", () => { clearTimeout(window.__pfr); window.__pfr = setTimeout(() => refresh(), 200); });
  // a stream that dies silently still gets a refresh every few seconds
  if (!window.__PF_STATIC__) setInterval(() => { if (!document.hidden) refresh(); }, 15000);

  get("/api/ping").then((p) => { document.getElementById("ws-name").textContent = p.workspace; }).catch(() => {});
  connect();
  route();
})();
