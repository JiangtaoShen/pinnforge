/* PINNForge discovery replay — a finished run, played back as a film.
 *
 * One stage, 1920 × 1080, scaled to the window. Everything on it is a pure
 * function of one clock T: the opening card, then each block in turn — what
 * the agent set out to do, the candidates it wrote and how they scored, the
 * papers it reached for, the notes it left, what it spent — then the closing
 * card. Seeking anywhere draws exactly what playing to that point would.
 *
 * The data is `/api/runs/<id>/story` (pinnforge/ui/story.py) and the archived
 * prediction frames, both read through the dashboard's own read-only API: no
 * reference values, no error field. The field panel shows each champion's
 * prediction, and beside it how much that champion changed the prediction
 * it replaced.
 *
 * Text that agents or task authors wrote goes in with textContent; equations
 * are escaped before the few tags this file adds for italics and scripts.
 */
"use strict";
(() => {
  const W = 1920, H = 1080;
  const OPEN = 5, CLOSE = 5, LEAD = 3.25, TAIL = 0.35;

  // ── motion: three helpers, nothing else eases ──
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = {
    outCubic: (t) => 1 - Math.pow(1 - t, 3),
    inOutQuad: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
    outBack: (t) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
  };
  const span = (t, start, dur, fn) => (t <= start ? 0 : t >= start + dur ? 1 : fn((t - start) / dur));
  const MOTION = {
    enter: (t, start, dur = 0.5) => span(t, start, dur, ease.outCubic),
    draw: (t, start, dur = 1) => span(t, start, dur, ease.inOutQuad),
    pop: (t, start, dur = 0.36) => span(t, start, dur, ease.outBack),
  };

  // ── small things ──
  const sci = (v) => (v == null ? "—" : v >= 0.1 ? v.toFixed(4) : v.toFixed(5));
  const kfmt = (v) => (v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : v >= 1e3 ? (v / 1e3).toFixed(0) + "k" : String(Math.round(v || 0)));
  const pad2 = (n) => String(n).padStart(2, "0");
  const sentence = (t) => { const x = String(t || ""); return x ? x.charAt(0).toUpperCase() + x.slice(1).toLowerCase() : x; };
  const dropEnum = (t) => String(t || "").replace(/^\s*(?:[-•*]|\(?[a-d]\)|\d+[.)])\s+/, "");
  const fmtDuration = (ms) => {
    if (!ms) return "—";
    const s = Math.round(ms / 1000), hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60);
    return hh ? `${hh}h${pad2(mm)}m` : mm ? `${mm}m${pad2(s % 60)}s` : `${s}s`;
  };
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // TeX → a few <i>/<sub>/<sup> tags around escaped text: variables italic,
  // digits, operators and function names upright. Every character of the
  // source is escaped first, so the tags are the only markup there is.
  const GREEK = { alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε",
    zeta: "ζ", eta: "η", theta: "θ", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π",
    rho: "ρ", sigma: "σ", tau: "τ", phi: "φ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
    Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω" };
  const SYMBOL = { partial: "∂", nabla: "∇", cdot: "·", times: "×", approx: "≈", in: "∈", infty: "∞",
    le: "≤", leq: "≤", ge: "≥", geq: "≥", ne: "≠", neq: "≠", pm: "±", to: "→", rightarrow: "→", ldots: "…", dots: "…",
    quad: " ", qquad: "  ", int: "∫", sum: "∑", prod: "∏" };
  function texToHtml(src) {
    if (!src) return "";
    const keep = [];
    const hold = (html) => "\u0001" + (keep.push(html) - 1) + "\u0002";
    let t = String(src).replace(/&/g, "\u0005").replace(/</g, "\u0003").replace(/>/g, "\u0004");
    t = t.replace(/\\[,;: !]|~/g, " ")
      .replace(/\\(sin|cos|tan|log|ln|exp|min|max|sinh|cosh|tanh)\b/g, (m, w) => hold(w))
      .replace(/\\(?:mathrm|text|operatorname|mathbf)\{([^}]*)\}/g, (m, w) => hold(w))
      .replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, "($1)/($2)")
      .replace(/\\sqrt\{([^}]*)\}/g, "√($1)")
      .replace(/\\(left|right|big|Big|bigg|Bigg)\b/g, "")
      .replace(/\\([A-Za-z]+)/g, (m, w) => (GREEK[w] ? GREEK[w] : SYMBOL[w] ? hold(SYMBOL[w]) : w))
      .replace(/ - /g, " − ");
    t = t.replace(/[A-Za-z]+|[α-ωΓΔΘΛΠΣΦΨΩ]/g, (m) => "<i>" + m + "</i>");
    t = t.replace(/_\{([^}]*)\}/g, "<sub>$1</sub>").replace(/\^\{([^}]*)\}/g, "<sup>$1</sup>")
      .replace(/_(<i>[^<]*<\/i>|[0-9])/g, "<sub>$1</sub>").replace(/\^(<i>[^<]*<\/i>|[0-9])/g, "<sup>$1</sup>")
      .replace(/[{}]/g, "");
    t = t.replace(/\u0001(\d+)\u0002/g, (m, i) => keep[+i]);
    return t.replace(/\u0005/g, "&amp;").replace(/\u0003/g, "&lt;").replace(/\u0004/g, "&gt;");
  }
  // inline `$…$` in prose → the same treatment, the prose around it escaped
  function proseToHtml(src) {
    const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return String(src || "").split(/\$([^$]+)\$/).map((part, k) => (k % 2 ? texToHtml(part) : esc(part))).join("");
  }

  // ── colour maps ──
  const HEX = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const STOPS = {
    turbo: [[0, "#30123B"], [0.07, "#4145AB"], [0.13, "#4675ED"], [0.2, "#39A2FC"], [0.27, "#1BCFD4"],
      [0.34, "#24ECA6"], [0.41, "#61FC6C"], [0.48, "#A4FC3B"], [0.55, "#D1E834"], [0.62, "#F3C63A"],
      [0.69, "#FE9B2D"], [0.76, "#F36315"], [0.83, "#D93807"], [0.9, "#B11901"], [1, "#7A0403"]],
    plume: [[0, "#FCFAF4"], [0.18, "#F2DFC6"], [0.4, "#E3B183"], [0.62, "#D97757"], [0.82, "#A6462A"], [1, "#521F0F"]],
    // the same ramp for the dark stage: nothing changed reads as the panel, not as a light
    plumeDark: [[0, "#1F1C18"], [0.18, "#3B2A1F"], [0.4, "#7A3F26"], [0.62, "#C0603E"], [0.82, "#E39A6B"], [1, "#F6DDB4"]],
  };
  function lut(name) {
    const stops = STOPS[name];
    const out = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let a = stops[0], b = stops[stops.length - 1];
      for (let s = 0; s < stops.length - 1; s++) if (t >= stops[s][0] && t <= stops[s + 1][0]) { a = stops[s]; b = stops[s + 1]; break; }
      const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
      const ca = HEX(a[1]), cb = HEX(b[1]);
      for (let c = 0; c < 3; c++) out[i * 3 + c] = ca[c] + (cb[c] - ca[c]) * f;
    }
    return out;
  }
  const gradient = (name) => `linear-gradient(90deg, ${STOPS[name].map(([p, h]) => `${h} ${(p * 100).toFixed(0)}%`).join(", ")})`;
  function heatColor(t, stops) {
    let a = stops[0], b = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) if (t >= stops[i][0] && t <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
    const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
    const ca = HEX(a[1]), cb = HEX(b[1]);
    return `rgb(${ca.map((v, i) => Math.round(v + (cb[i] - v) * f)).join(",")})`;
  }
  const HEAT = {
    light: [[0, "#F0DFC6"], [0.4, "#E1A377"], [0.7, "#CB6A42"], [1, "#8B3A1C"]],
    dark: [[0, "#4A2A1E"], [0.45, "#A9502F"], [0.75, "#D97757"], [1, "#E8C08A"]],
  };

  // ── marks (from the design's supplied assets) ──
  const SVGNS = "http://www.w3.org/2000/svg";
  function svgMark(markup, size) {
    const tpl = document.createElement("template");
    tpl.innerHTML = markup.trim();  // constant markup, defined below
    const svg = tpl.content.firstChild;
    svg.setAttribute("width", size); svg.setAttribute("height", size);
    return svg;
  }
  const AGENT_SVG = `<svg viewBox="0 0 60 60" class="rp-agent-mark"><g class="m-dark"><path d="m10 34h6v7h-6z"/><path d="m44 34h6v7h-6z"/><path d="m26 26h8v6h-8z"/><path d="m13 8h2v12h-2c-2.208 0-4-1.792-4-4v-4c0-2.208 1.792-4 4-4z"/><path d="m49 8h2v12h-2c-2.208 0-4-1.792-4-4v-4c0-2.208 1.792-4 4-4z" transform="matrix(-1 0 0 -1 96 28)"/></g><rect class="m-light" height="26" rx="7" width="34" x="13" y="1"/><rect fill="#2A2620" height="26" rx="5" transform="matrix(0 1 -1 0 44 -16)" width="16" x="22" y="1"/><circle class="m-eye" cx="24" cy="13" r="3"/><circle class="m-eye" cx="36" cy="13" r="3"/><path class="m-light" d="m18 31h24c1.656 0 3 1.344 3 3v10c0 8.279-6.721 15-15 15-8.279 0-15-6.721-15-15v-10c0-1.656 1.344-3 3-3z"/><path class="m-dark" d="m11 52v2.5c0 1.38-1.12 2.5-2.5 2.5s-2.5-1.12-2.5-2.5v-8.5z"/><path class="m-light" d="m11 33v19s-5 0-5-6v-10c0-1.1.45-2.1 1.17-2.83.73-.72 1.73-1.17 2.83-1.17.55 0 1 .45 1 1z"/><path class="m-dark" d="m54 46v8.5c0 1.38-1.12 2.5-2.5 2.5s-2.5-1.12-2.5-2.5v-2.5z"/><path class="m-light" d="m54 36v10c0 6-5 6-5 6v-19c0-.55.45-1 1-1 1.1 0 2.1.45 2.83 1.17.72.73 1.17 1.73 1.17 2.83z"/><rect class="m-eye" height="10" rx="1" width="6" x="27" y="36"/><path class="m-eye" d="m31 19h-2c-.553 0-1-.447-1-1s.447-1 1-1h2c.553 0 1 .447 1 1s-.447 1-1 1z"/></svg>`;
  const FORGE_SVG = `<svg viewBox="0 0 512 512" class="rp-forge-mark"><g clip-rule="evenodd" fill-rule="evenodd"><path d="m166.545 335.111h105.785v137.872h-105.785z" fill="#5C554A"/><path d="m210.625 220.034h276.8c1.403 0 2.611.511 3.588 1.518.976 1.006 1.454 2.228 1.412 3.629-2.496 86.252-67.898 113.881-127.570 113.881h-154.23c-2.754 0-5-2.246-5-5v-109.028c0-2.754 2.246-5 5-5z" fill="#5C554A"/><g fill="#9A9080"><path d="m17.5 188.128h308.817c2.754 0 5 2.246 5 5v126.983c0 2.754-2.246 5-5 5h-308.817c-2.754 0-5-2.246-5-5v-126.983c0-2.754 2.246-5 5-5z"/><path d="m112.559 255.917h213.758c2.754 0 5 2.246 5 5v93.293c0 2.754-2.246 5-5 5h-213.758c-2.754 0-5-2.246-5-5v-93.293c0-2.754 2.246-5 5-5z"/><path d="m113.777 448.141h211.322c13.007 0 23.649 10.642 23.649 23.649v18.649c0 2.754-2.246 5-5 5h-248.62c-2.754 0-5-2.246-5-5v-18.649c0-13.008 10.642-23.649 23.649-23.649z"/></g><path d="m369.425 80.281 4.87 18.177c.713 2.66-.875 5.411-3.535 6.123l-193.758 51.917c-2.66.713-5.411-.875-6.123-3.535l-4.87-18.177c-.713-2.66.875-5.411 3.535-6.123l193.757-51.917c2.661-.713 5.412.875 6.124 3.535z" fill="#d5a889"/><path d="m391.215 20.268 31.958 119.268c.713 2.66-.875 5.411-3.535 6.123l-62.822 16.833c-2.66.713-5.411-.875-6.123-3.535l-31.959-119.268c-.713-2.66.875-5.411 3.535-6.123l62.822-16.833c2.661-.713 5.412.875 6.124 3.535z" fill="#5C554A"/><path d="m380.584 191.092c-3.866 0-7-3.134-7-7s3.134-7 7-7h111.916c3.866 0 7 3.134 7 7s-3.134 7-7 7z" fill="#C8603C"/><path d="m445.754 88.999c0 3.866-3.134 7-7 7s-7-3.134-7-7v-45.519c0-3.866 3.134-7 7-7s7 3.134 7 7zm1.227 62.366c-3.866 0-7-3.134-7-7s3.134-7 7-7h45.519c3.866 0 7 3.134 7 7s-3.134 7-7 7zm4.949-28.279 32.187-32.186c2.733-2.733 2.733-7.165 0-9.899-2.733-2.733-7.165-2.733-9.899 0l-32.187 32.187c-2.733 2.733-2.733 7.165 0 9.899 2.734 2.732 7.166 2.732 9.899-.001z" fill="#B98432"/></g></svg>`;
  const ICE_SVG = `<svg viewBox="0 0 512 512" class="rp-ice"><path d="m399.028 128-143.028 128 78.674 210.577 143.029-82.577v-256z" fill="#7C9AA8"/><path d="m334.674 282.845c0 24.770 26.814 40.251 48.266 27.866 9.956-5.748 16.089-16.37 16.089-27.866v-154.845h-128.709z" fill="#9FB6C0"/><path d="m334.674 128h-223.471l144.797 384 78.674-45.423z" fill="#8FA9B5"/><path d="m241.68 128h-207.383v256l143.029 82.577z" fill="#B8C7CD"/><path d="m112.972 128v173.423l64.354 37.154v128l78.674 45.423v-384z" fill="#9FB6C0"/><path d="m256 0-64.354 128 64.354 128 221.703-128z" fill="#B8C7CD"/><path d="m34.297 128 221.703 128v-256z" fill="#D2DBDD"/></svg>`;
  const FLAME_SVG = `<svg viewBox="0 0 68 68" class="rp-flame-svg"><path class="f-outer" d="m29.7 65.9c-7.2-.7-14.5-4.2-18.1-10.5-3.5-6.3-2.6-14.3-.1-21.2l2.1 6c.2.6.5 1.3 1 1.6.8.5 1.9.1 2.5-.7.5-.8.6-1.8.5-2.7-.3-4-2.6-7.7-2.7-11.7-.1-4.6 2.8-8.8 6.4-11.7.1 1.6.5 3.3 1 4.8.2.6.6 1.2 1.2 1.3.4 0 .8-.2 1.1-.5 3-2.5 4.8-6.1 5.5-9.9.3-1.9.3-3.9-.1-5.8-.1-.4-.9-2.4-.7-2.7.3-.6 2 .7 2.4.9 6.6 4.2 11 11.7 11.1 19.5 0 1.2.2 2.8 1.4 3 .7.1 1.4-.3 1.8-.9s.4-1.3.5-2c4.7 4.2 8.1 9.4 10.4 15.2 1.3 3.4 2.2 7.1 1.5 10.7-.6 3.3-2.2 6.3-4.5 8.8-3.8 4.3-8.9 7.7-14.6 8.3-3 .3-6.2.4-9.6.2z" fill="#CF5F38"/><path class="f-inner" d="m47.3 61.3c-.2.7-.5 1.5-.8 2.2-2.2 1.2-4.6 2-7.1 2.2-1.9.2-3.8.3-5.8.3-1.2 0-2.5 0-3.8-.1-2.3-.2-4.6-.7-6.8-1.5-1.2-1.7-2-3.7-2.3-5.7-.7-4.4.6-8.9 3.1-12.6.4-.6 1.3-.4 1.5.3.7 2.3 2.2 4.4 3.9 6.2.5.5 1.3.2 1.4-.5.2-2.4.7-6.9 1.6-9.3 1.1-3 2.8-5.9 5.3-8 .5-.4 1.2-.2 1.3.5.5 3 2 5.8 3.5 8.5 1.8 3 3.8 6 4.7 9.4.9 2.5 1 5.4.3 8.1z" fill="#D19B45"/></svg>`;

  // ── the story, prepared ──
  function prep(story) {
    const blocks = story.blocks.length ? story.blocks : [];
    const heat = {};
    const out = blocks.map((b) => {
      for (const c of b.cited || []) heat[c.slot] = (heat[c.slot] || 0) + 1;
      return { ...b, heatAfter: { ...heat } };
    });
    const practice = [];
    out.forEach((b, i) => (b.notes || []).forEach((p, k) => practice.push({ ...p, bi: i, ord: k, n: b.notes.length })));
    const scores = [];
    out.forEach((b) => { (b.roster || []).forEach((r) => r.r != null && scores.push(r.r)); if (b.exit != null) scores.push(b.exit); });
    return {
      ...story, blocks: out, practice,
      maxTok: Math.max(1, ...out.map((b) => b.cost.tokens || 0)),
      scoreMin: Math.min(...(scores.length ? scores : [1])),
      scoreMax: Math.max(story.baseline || 0, ...(scores.length ? scores : [1])),
    };
  }

  // seconds per block on screen: a long run plays briskly, a short one lingers
  const perBlock = (n) => clamp(14.4 / Math.max(1, n), 1.3, 3.2);
  function timeline(n) {
    const caseDur = LEAD + perBlock(n) * n + TAIL;
    return { caseStart: OPEN, caseEnd: OPEN + caseDur, closing: OPEN + caseDur, total: OPEN + caseDur + CLOSE };
  }
  function cursor(T, tl, d) {
    const NB = d.blocks.length, local = T - tl.caseStart, per = perBlock(NB);
    const raw = (local - LEAD) / per;
    const i = clamp(Math.floor(raw), 0, NB - 1);
    const ph = clamp(raw - i, 0, 1);
    const b = d.blocks[i], prev = d.blocks[Math.max(0, i - 1)];
    const phase = local < LEAD ? "init" : ph < 0.22 ? "thinking" : ph < 0.42 ? "retrieving" : ph < 0.7 ? "executing" : ph < 0.86 ? "evaluating" : "writing";
    const lerp = (a, z, t) => (a || 0) + ((z || 0) - (a || 0)) * t;
    const entry = b.entry != null ? b.entry : b.exit;
    return {
      i, ph, phase, b, prev, local, started: local >= LEAD,
      tokens: lerp(i === 0 ? 0 : prev.cost.tok_cum, b.cost.tok_cum, clamp(ph / 0.9, 0, 1)),
      gpu: lerp(i === 0 ? 0 : prev.cost.gpu_cum_h, b.cost.gpu_cum_h, clamp(ph / 0.9, 0, 1)),
      score: lerp(entry, b.exit, clamp((ph - 0.66) / 0.2, 0, 1)),
      tools: lerp(0, b.cost.tools || 0, ph),
    };
  }

  // ── panel chrome ──
  function panel(title, right, w, h, opts = {}) {
    const root = el("div", `rp-panel${opts.bare ? " bare" : ""}`);
    root.style.width = `${w}px`; root.style.height = `${h}px`;
    if (!opts.bare) root.append(el("i", "rp-tick bl"), el("i", "rp-tick br"));
    const head = el("div", "rp-panel-head");
    const t = el("div", "rp-panel-title");
    if (typeof title === "string") t.textContent = title; else t.append(...title);
    const r = el("div", "rp-panel-right");
    head.append(t, r);
    const body = el("div", "rp-panel-body");
    root.append(head, body);
    const setRight = (items) => r.replaceChildren(...items.filter(Boolean).map((x) => (typeof x === "string" ? el("span", "rp-chip", x) : x)));
    setRight(right || []);
    return { root, body, setRight };
  }

  // ───────────────────────────── the replay ─────────────────────────────
  class Replay {
    constructor(host, runId) {
      this.host = host;
      this.runId = runId;
      this.T = 0;
      this.playing = false;
      this.raf = 0;
      this.dead = false;
    }

    async load() {
      const PF = window.PF;
      const story = await PF.get(`/api/runs/${encodeURIComponent(this.runId)}/story`);
      if (this.dead) return;
      this.d = prep(story);
      this.tl = timeline(this.d.blocks.length);
      this.theme = document.documentElement.getAttribute("data-theme") ||
        (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      this.fields = await this.loadFields();
      if (this.dead) return;
      this.build();
      this.seek(0);
      if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) this.play();
    }

    destroy() {
      this.dead = true;
      cancelAnimationFrame(this.raf);
      document.removeEventListener("keydown", this.onKey);
      window.removeEventListener("resize", this.onResize);
    }

    // ── prediction frames → images, once ──
    async loadFields() {
      const PF = window.PF, d = this.d;
      const ids = [...new Set(d.blocks.map((b) => b.frame).filter((f) => f != null))];
      if (!ids.length || !d.grid) return null;
      const data = {};
      for (const n of ids) {
        try { data[n] = await PF.get(`/api/runs/${encodeURIComponent(this.runId)}/frame/${n}`); } catch { /* skipped */ }
        if (this.dead) return null;
      }
      const loaded = ids.filter((n) => data[n]);
      if (!loaded.length) return null;
      const view = PF.field.resolveView({ grid: d.grid, view: d.view });
      const coords = (d.grid.coords || []).slice(0, 3);
      const first = data[loaded[0]];
      const cols = first.coords;
      const xi = coords.indexOf(view.axes.x), yi = view.axes.y != null ? coords.indexOf(view.axes.y) : -1;
      if (xi < 0) return null;
      const axis = view.time || (view.slice && view.slice.axis) || null;
      const si = axis ? coords.indexOf(axis) : -1;
      const all = cols[0].map((_, i) => i);
      let tiles;
      if (si >= 0) {
        const steps = [...new Set(cols[si])].sort((a, b) => a - b);
        const k = Math.min(4, steps.length);
        const picks = [...new Set(Array.from({ length: k }, (_, j) => Math.round((j * (steps.length - 1)) / Math.max(1, k - 1))))];
        tiles = picks.map((p) => ({ label: `${axis} = ${Number(steps[p].toPrecision(3))}`, rows: all.filter((i) => cols[si][i] === steps[p]), panel: view.panels[0] }));
      } else {
        tiles = view.panels.slice(0, 4).map((p) => ({ label: p.title, rows: all, panel: p }));
      }
      const X = cols[xi], Y = yi >= 0 ? cols[yi] : null;
      let aspect = null;
      if (Y && !(/^t(ime)?$/i.test(view.axes.y || "") || /^t(ime)?$/i.test(view.axes.x || ""))) {
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (let i = 0; i < X.length; i++) {
          if (X[i] < x0) x0 = X[i]; if (X[i] > x1) x1 = X[i];
          if (Y[i] < y0) y0 = Y[i]; if (Y[i] > y1) y1 = Y[i];
        }
        aspect = clamp((y1 - y0) / Math.max(1e-12, x1 - x0), 0.25, 1.6);
      }
      // values per frame per tile, a shared range per tile, and the change each champion made
      const vals = {};
      for (const n of loaded) vals[n] = tiles.map((t) => PF.field.panelValues(t.panel.show, data[n].fields));
      const ranges = tiles.map((t, k) => {
        let lo = Infinity, hi = -Infinity;
        for (const n of loaded) { const v = vals[n][k]; if (!v) continue; for (const i of t.rows) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; } }
        if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
        const div = lo < 0 && hi > 0 && Math.min(-lo, hi) >= 0.2 * Math.max(-lo, hi);
        if (div) { const a = Math.max(-lo, hi); lo = -a; hi = a; }
        return { lo, hi: hi === lo ? lo + 1 : hi };
      });
      // keyed by block: a block that did not beat the champion changed nothing
      const order = d.blocks.map((b) => (data[b.frame] ? b.frame : null));
      const diffs = {};
      const mags = [];
      for (let bi = 1; bi < order.length; bi++) {
        const a = order[bi - 1], b = order[bi];
        if (a == null || b == null || a === b) continue;
        diffs[bi] = tiles.map((t, k) => {
          const va = vals[a][k], vb = vals[b][k];
          if (!va || !vb) return null;
          const out = new Float64Array(va.length);
          for (const i of t.rows) { out[i] = Math.abs(vb[i] - va[i]); mags.push(out[i]); }
          return out;
        });
      }
      mags.sort((p, q) => p - q);
      const dmax = mags.length ? Math.max(1e-12, mags[Math.floor(0.99 * (mags.length - 1))]) : 1;
      return { tiles, X, Y, aspect, vals, ranges, diffs, dmax, order, quantity: (tiles[0] && tiles[0].panel.title) || "u" };
    }

    // ── building the stage ──
    build() {
      const d = this.d;
      this.shell = el("div", "rp-shell");
      this.viewport = el("div", "rp-viewport");
      this.stage = el("div", `rp-stage rp-${this.theme}`);
      this.viewport.append(this.stage);
      this.shell.append(this.viewport, this.buildBar());
      this.host.replaceChildren(this.shell);

      this.stage.append(el("div", "rp-glow"));
      this.dash = el("div", "rp-dash");
      this.stage.append(this.dash);
      const GAP = 16, colA = 430, colB = 806, colC = 612;
      this.header = this.buildHeader();
      const row = el("div", "rp-row");
      const left = el("div", "rp-col");
      const top = el("div", "rp-row-in");
      const bottom = el("div", "rp-row-in");
      this.eq = this.buildEq(colA, 320);
      this.agent = this.buildAgent(colB, 320);
      this.field = this.buildFields(colB, 626);
      const stack = el("div", "rp-col");
      stack.style.width = `${colA}px`;
      this.conv = this.buildConv(colA, 310);
      this.cost = this.buildCost(colA, 300);
      stack.append(this.conv.root, this.cost.root);
      top.append(this.eq.root, this.agent.root);
      bottom.append(this.field.root, stack);
      left.append(top, bottom);
      const kb = el("div", "rp-kb");
      kb.style.width = `${colC}px`;
      kb.append(el("i", "rp-tick tl"), el("i", "rp-tick tr"), el("i", "rp-tick bl"), el("i", "rp-tick br"));
      this.kb1 = this.buildKB1(colC, 372);
      this.kb2 = this.buildKB2(colC, 587);
      kb.append(this.kb1.root, el("div", "rp-rule"), this.kb2.root);
      row.append(left, kb);
      row.style.gap = left.style.gap = top.style.gap = bottom.style.gap = stack.style.gap = `${GAP}px`;
      this.kbBox = kb;
      this.dash.append(this.header.root, row);
      this.opening = this.buildCover(false);
      this.closing = this.buildCover(true);
      this.stage.append(this.opening.root, this.closing.root);
      this.panels = [[this.eq, 0.55], [this.agent, 0.51], [this.field, 0.35], [this.conv, 0.47], [this.cost, 0.39]];
      this.onResize = () => this.fit();
      window.addEventListener("resize", this.onResize);
      this.onKey = (e) => {
        if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
        if (e.key === " ") { e.preventDefault(); this.toggle(); }
        else if (e.key === "ArrowRight") this.seek(this.T + 2);
        else if (e.key === "ArrowLeft") this.seek(this.T - 2);
      };
      document.addEventListener("keydown", this.onKey);
      this.fit();
      this.lastI = -1;
    }

    buildBar() {
      const bar = el("div", "rp-bar");
      this.playBtn = el("button", "rp-btn", "▶");
      this.playBtn.type = "button";
      this.playBtn.setAttribute("aria-label", "play");
      this.playBtn.onclick = () => this.toggle();
      const restart = el("button", "rp-btn", "⟲");
      restart.type = "button";
      restart.setAttribute("aria-label", "restart");
      restart.onclick = () => { this.seek(0); this.play(); };
      this.clock = el("span", "rp-time", "0:00.0");
      this.scrub = el("input", "rp-scrub");
      Object.assign(this.scrub, { type: "range", min: "0", max: String(Math.round(this.tl.total * 100)), value: "0" });
      this.scrub.setAttribute("aria-label", "time");
      this.scrub.oninput = () => { this.pause(); this.seek(Number(this.scrub.value) / 100); };
      const total = el("span", "rp-time", this.fmtTime(this.tl.total));
      const full = el("button", "rp-btn", "⛶");
      full.type = "button";
      full.setAttribute("aria-label", "full screen");
      full.onclick = () => {
        const p = document.fullscreenElement ? document.exitFullscreen()
          : this.shell.requestFullscreen ? this.shell.requestFullscreen() : null;
        if (p) p.catch(() => {});
      };
      bar.append(this.playBtn, restart, this.clock, this.scrub, total, full);
      return bar;
    }
    fmtTime(t) { const m = Math.floor(t / 60), s = t - m * 60; return `${m}:${s.toFixed(1).padStart(4, "0")}`; }

    fit() {
      const box = this.viewport.getBoundingClientRect();
      const k = Math.min(box.width / W, (box.height || box.width * (H / W)) / H);
      this.stage.style.transform = `scale(${k})`;
      this.stage.style.left = `${(box.width - W * k) / 2}px`;
      this.stage.style.top = `${Math.max(0, (box.height - H * k) / 2)}px`;
    }

    buildHeader() {
      const d = this.d, root = el("div", "rp-header");
      const brand = el("div", "rp-brand");
      const words = el("div");
      words.append(el("div", "rp-brand-name", "PINNForge"), el("div", "rp-brand-sub", "Autonomous PINN discovery"));
      brand.append(svgMark(FORGE_SVG, 42), words);
      const mid = el("div", "rp-header-mid");
      const badge = el("div", "rp-badge");
      const sec = d.train_s ? `${Math.round(d.train_s)} s` : "—";
      const gpu = el("img", "rp-gpu");
      gpu.alt = "";
      gpu.src = (window.__PF_ASSETS__ && window.__PF_ASSETS__.gpu) || "gpu.svg";
      badge.append(gpu, el("span", "rp-badge-sep"), el("span", "rp-badge-text", d.cpu ? `CPU · ${sec}` : sec));
      mid.append(el("div", "rp-case-tag", String(d.run)), el("div", "rp-case-name", d.name || d.task), badge);
      const right = el("div", "rp-header-right");
      const logo = el("img", "rp-logo");
      logo.alt = "COLALab";
      logo.src = (window.__PF_ASSETS__ && window.__PF_ASSETS__.colalab) || "colalab.png";
      const logoBox = el("div", "rp-logo-box");
      logoBox.append(logo);
      const status = el("div", "rp-status");
      right.append(status, logoBox);
      root.append(brand, mid, right);
      return { root, status };
    }

    buildEq(w, h) {
      const d = this.d;
      const p = panel("PDE", ["Relative L² error", d.cpu ? "CPU" : "GPU"], w, h);
      const list = el("div", "rp-eqs");
      this.eqRows = (d.equations || []).map((e) => {
        const row = el("div", "rp-eq");
        if (e.name) { const lab = el("div", "rp-label"); lab.innerHTML = proseToHtml(e.name); row.append(lab); }
        const math = el("div", `rp-math${d.equations.length > 3 ? " small" : ""}`);
        math.innerHTML = texToHtml(e.tex);
        row.append(math);
        list.append(row);
        return row;
      });
      if (!this.eqRows.length) list.append(el("div", "rp-faint", "problem.md states no displayed equation"));
      const facts = el("div", "rp-facts");
      for (const [k, v] of d.facts || []) {
        const r = el("div", "rp-fact");
        const val = el("span", "rp-fact-v");
        val.innerHTML = proseToHtml(v);
        r.append(el("span", "rp-fact-k", k), val);
        facts.append(r);
      }
      const rule = el("div", "rp-rule");
      p.body.append(list, rule, facts);
      return { ...p, rule, facts };
    }

    buildAgent(w, h) {
      const p = panel("Agent session", [], w, h);
      const wrap = el("div", "rp-agent");
      const side = el("div", "rp-agent-side");
      this.mark = el("div", "rp-mark");
      this.ring = el("div", "rp-ring");
      this.mark.append(this.ring, svgMark(AGENT_SVG, 46));
      this.phaseEl = el("div", "rp-phase", "Init");
      side.append(this.mark, this.phaseEl, el("div", "rp-stem"));
      const main = el("div", "rp-agent-main");
      this.headline = el("div", "rp-headline");
      this.thrust = el("div", "rp-thrust");
      this.tags = el("div", "rp-tags");
      this.quote = el("div", "rp-quote");
      main.append(this.headline, this.thrust, this.tags, this.quote);
      const roster = el("div", "rp-roster");
      this.rosterList = el("div", "rp-roster-list");
      this.outcomes = el("div", "rp-outcomes");
      roster.append(el("div", "rp-label", "Candidate roster"), this.rosterList, this.outcomes);
      wrap.append(side, main, roster);
      p.body.append(wrap);
      return p;
    }

    buildFields(w, h) {
      const p = panel("Physical field", [], w, h);
      const f = this.fields;
      const wrap = el("div", "rp-fields");
      if (!f) {
        wrap.append(el("div", "rp-empty", "No prediction was archived for this run."));
        p.body.append(wrap);
        return { ...p, cols: [] };
      }
      const slicing = f.tiles.length && /=/.test(f.tiles[0].label);
      p.setRight([`${f.tiles.length} ${slicing ? (/^t(ime)?\s*=/i.test(f.tiles[0].label) ? "time" : "span") : "field"} slice${f.tiles.length === 1 ? "" : "s"}`]);
      const BOXW = (w - 28 - 33) / 2, availH = h - 78 - 24 - 34;
      const N = f.tiles.length, GAPT = 12;
      const ar = f.aspect ? 1 / f.aspect : 1.2;  // width over height
      let best = null;
      for (let cols = 1; cols <= N; cols++) {
        const rows = Math.ceil(N / cols);
        const wMax = (BOXW - (cols - 1) * GAPT) / cols, hMax = (availH - (rows - 1) * GAPT) / rows;
        const th = Math.floor(Math.min(hMax, wMax / ar));
        if (th < 40) continue;
        const area = th * th * ar;
        if (!best || area > best.area) best = { cols, th, tw: Math.round(th * ar), area };
      }
      const fit = best || { cols: 1, th: 120, tw: 160 };
      const column = (kind, title) => {
        const col = el("div", "rp-field-col");
        col.style.width = `${BOXW}px`;
        col.append(el("div", "rp-label strong", title));
        const grid = el("div", "rp-tiles");
        grid.style.gridTemplateColumns = `repeat(${fit.cols}, ${fit.tw}px)`;
        grid.style.gap = `${GAPT}px`;
        const tiles = f.tiles.map((t) => {
          const tile = el("div", "rp-tile");
          tile.style.width = `${fit.tw}px`; tile.style.height = `${fit.th}px`;
          const note = el("div", "rp-tile-note");
          tile.append(el("div", "rp-tile-label", t.label), note);
          grid.append(tile);
          return { tile, layers: {}, note };
        });
        col.append(grid);
        const bar = el("div", "rp-cbar");
        const strip = el("div", "rp-cbar-strip");
        strip.style.background = gradient(kind === "pred" ? "turbo" : this.plumeName());
        const labels = el("div", "rp-cbar-labels");
        const k = kind === "pred" ? f.ranges[0] : { lo: 0, hi: f.dmax };
        labels.append(el("span", null, Number(k.lo.toPrecision(3)).toString()),
          el("span", null, kind === "pred" ? f.quantity : `|Δ ${f.quantity}|`), el("span", null, Number(k.hi.toPrecision(3)).toString()));
        bar.append(strip, labels);
        col.append(bar);
        return { col, tiles };
      };
      const pred = column("pred", "Prediction");
      const diff = column("diff", "Change made by the incumbent");
      const sep = el("div", "rp-vsep");
      wrap.append(pred.col, sep, diff.col);
      p.body.append(wrap);
      this.tileSize = { w: fit.tw, h: fit.th };
      return { ...p, cols: [pred, diff] };
    }

    plumeName() { return this.theme === "dark" ? "plumeDark" : "plume"; }

    // one canvas per (frame, tile), drawn the first time it is shown
    layer(colIdx, k, key, draw) {
      const tile = this.field.cols[colIdx].tiles[k];
      if (!tile.layers[key]) {
        const c = document.createElement("canvas");
        const dpr = 1.5;
        c.width = Math.round(this.tileSize.w * dpr); c.height = Math.round(this.tileSize.h * dpr);
        c.className = "rp-layer";
        draw(c.getContext("2d"), c.width, c.height);
        tile.tile.insertBefore(c, tile.tile.firstChild);
        tile.layers[key] = c;
      }
      for (const [k2, c] of Object.entries(tile.layers)) c.style.opacity = k2 === key ? "1" : "0";
    }

    showFields(i) {
      const f = this.fields, PF = window.PF;
      if (!f) return;
      const frame = f.order[i];
      const turbo = this.lutTurbo || (this.lutTurbo = lut("turbo"));
      const plume = this.lutPlume || (this.lutPlume = lut(this.plumeName()));
      const [predCol, diffCol] = this.field.cols;
      f.tiles.forEach((t, k) => {
        const r = f.ranges[k];
        const dv = frame != null && f.diffs[i] && f.diffs[i][k];
        predCol.tiles[k].note.textContent = frame == null ? "No prediction yet" : "";
        diffCol.tiles[k].note.textContent = frame == null ? "No prediction yet"
          : dv ? "" : (i === 0 || f.order[i - 1] == null) ? "First incumbent" : "Incumbent held";
        const paint = (vals, lo, hi, L) => (ctx, Wp, Hp) => {
          if (!vals) return;
          if (f.Y) {
            const ok = PF.field.drawTensor(ctx, t.rows, f.X, f.Y, vals, lo, hi, L, Wp, Hp);
            if (!ok) PF.field.drawSplat(ctx, t.rows, f.X, f.Y, vals, lo, hi, L, Wp, Hp);
          } else PF.field.drawLine(ctx, t.rows, f.X, vals, lo, hi, Wp, Hp, 1.5);
        };
        if (frame == null) { this.layer(0, k, "none", () => {}); this.layer(1, k, "none", () => {}); return; }
        this.layer(0, k, `f${frame}`, paint(f.vals[frame][k], r.lo, r.hi, turbo));
        this.layer(1, k, dv ? `d${i}` : "none", dv ? paint(dv, 0, f.dmax, plume) : () => {});
      });
    }

    buildConv(w, h) {
      const p = panel("Convergence", [], w, h, {});
      const top = el("div", "rp-conv-top");
      this.champ = el("span", "rp-big", "—");
      top.append(el("span", "rp-label strong", "Incumbent error"), this.champ);
      this.convSvg = document.createElementNS(SVGNS, "svg");
      this.convSvg.setAttribute("class", "rp-conv");
      this.convW = w - 28; this.convH = h - 78 - 20;
      this.convSvg.setAttribute("width", this.convW); this.convSvg.setAttribute("height", this.convH);
      p.body.append(top, this.convSvg);
      return p;
    }

    drawConv(cur) {
      const d = this.d, NB = d.blocks.length;
      const key = `${cur.i}|${Math.round(cur.ph * 40)}`;
      if (key === this.convKey) return;
      this.convKey = key;
      const PW = this.convW, PH = this.convH;
      const gx = 62, gy = 10, gw = PW - gx - 16, gh = PH - gy - 34;
      const lo = Math.log10(Math.max(d.scoreMin * 0.7, 1e-6)), hi = Math.log10(Math.max(d.scoreMax * 1.4, d.scoreMin * 10));
      const X = (i) => gx + (gw * i) / Math.max(1, NB - 1);
      const Y = (v) => gy + gh * (1 - (Math.log10(Math.max(v, 1e-9)) - lo) / (hi - lo));
      const parts = [];
      for (let dd = Math.ceil(lo); dd <= Math.floor(hi); dd++) {
        parts.push(`<line class="g" x1="${gx}" x2="${gx + gw}" y1="${Y(10 ** dd)}" y2="${Y(10 ** dd)}"/>`,
          `<text class="t" x="${gx - 7}" y="${Y(10 ** dd) + 4}" text-anchor="end">1e${dd}</text>`);
      }
      parts.push(`<line class="a" x1="${gx}" x2="${gx}" y1="${gy - 2}" y2="${gy + gh}"/>`,
        `<line class="a" x1="${gx}" x2="${gx + gw + 4}" y1="${gy + gh}" y2="${gy + gh}"/>`);
      if (d.baseline) {
        parts.push(`<line class="base" x1="${gx}" x2="${gx + gw}" y1="${Y(d.baseline)}" y2="${Y(d.baseline)}"/>`,
          `<text class="t base-t" x="${gx + gw}" y="${Y(d.baseline) - 6}" text-anchor="end">anchor</text>`);
      }
      for (let i = 0; i <= cur.i; i++) {
        const vis = i === cur.i ? clamp((cur.ph - 0.2) / 0.5, 0, 1) : 1;
        const rs = d.blocks[i].roster || [];
        rs.forEach((r, k) => {
          if (r.r == null || k / Math.max(rs.length, 1) > vis) return;
          parts.push(`<circle class="c ${r.st}${i === cur.i ? " now" : ""}" cx="${X(i) + ((k % 5) - 2) * 2.4}" cy="${Y(r.r)}" r="${i === cur.i ? 3 : 2}"/>`);
        });
      }
      const pts = [];
      for (let i = 0; i <= cur.i; i++) {
        const v = i === cur.i ? cur.score : d.blocks[i].exit;
        if (v != null) pts.push([X(i), Y(v)]);
      }
      if (pts.length > 1) parts.push(`<polyline class="champ" points="${pts.map((q) => q.join(",")).join(" ")}"/>`);
      pts.forEach((q) => parts.push(`<circle class="cp" cx="${q[0]}" cy="${q[1]}" r="2.6"/>`));
      if (pts.length) { const q = pts[pts.length - 1]; parts.push(`<circle class="head" cx="${q[0]}" cy="${q[1]}" r="5"/>`); }
      d.blocks.forEach((b, i) => parts.push(`<text class="t${i === cur.i ? " on" : ""}" x="${X(i)}" y="${gy + gh + 16}" text-anchor="middle">${pad2(i)}</text>`));
      parts.push(`<text class="t axis" x="${gx + gw / 2}" y="${gy + gh + 32}" text-anchor="middle">Block</text>`,
        `<text class="t axis" transform="translate(12 ${gy + gh / 2}) rotate(-90)" text-anchor="middle" dominant-baseline="central">Relative L² error</text>`);
      this.convSvg.innerHTML = parts.join("");  // numbers and fixed words only
    }

    buildCost(w, h) {
      const p = panel("Compute ledger", [], w, h);
      const wrap = el("div", "rp-cost");
      const g = el("div");
      const gRow = el("div", "rp-cost-row");
      this.gpuText = el("span", "rp-num");
      gRow.append(el("span", "rp-label strong", this.d.cpu ? "CPU-hours" : "GPU-hours"), this.gpuText);
      const meter = el("div", "rp-meter");
      this.gpuFill = el("div", "rp-meter-fill");
      meter.append(this.gpuFill, ...[0.25, 0.5, 0.75].map((f) => { const t = el("i", "rp-meter-tick"); t.style.left = `${f * 100}%`; return t; }));
      g.append(gRow, meter);
      const t = el("div");
      const tRow = el("div", "rp-cost-row");
      this.tokText = el("span", "rp-num");
      tRow.append(el("span", "rp-label strong", "Tokens"), this.tokText);
      this.tokBars = el("div", "rp-tok-bars");
      this.bars = this.d.blocks.map(() => { const b = el("div", "rp-tok-bar"); this.tokBars.append(b); return b; });
      t.append(tRow, this.tokBars);
      const counts = el("div", "rp-counts");
      this.countEls = ["tools", "cands", "evals", "smoke"].map((k) => {
        const c = el("div", "rp-count");
        const v = el("div", "rp-count-v", "—");
        c.append(el("div", "rp-label", sentence(k)), v);
        counts.append(c);
        return v;
      });
      wrap.append(g, t, counts);
      p.body.append(wrap);
      return p;
    }

    buildKB1(w, h) {
      const d = this.d;
      const title = [el("span", null, "Literature base"), svgMark(ICE_SVG, 19)];
      const p = panel(title, [], w, h, { bare: true });
      const COLS = 20;
      const cell = Math.min(23, Math.floor((w - 28 - (COLS - 1) * 3) / COLS));
      const grid = el("div", "rp-kb1-grid");
      grid.style.gridTemplateColumns = `repeat(${COLS}, ${cell}px)`;
      this.cells = {};
      for (const e of d.kb1.entries) {
        const c = el("div", "rp-cell");
        c.style.width = c.style.height = `${cell}px`;
        grid.append(c);
        this.cells[e.s] = c;
      }
      this.citedList = el("div", "rp-cited");
      p.body.append(grid, el("div", "rp-rule"), this.citedList);
      if (!d.kb1.entries.length) grid.append(el("div", "rp-faint", "the run carries no theory corpus"));
      return p;
    }

    buildKB2(w, h) {
      const title = [el("span", null, "Experience base"), (() => { const f = el("span", "rp-flame"); f.append(el("span", "rp-flame-glow"), svgMark(FLAME_SVG, 19)); return f; })()];
      const p = panel(title, [], w, h, { bare: true });
      const top = el("div", "rp-kb2-top");
      this.kb2Count = el("span", "rp-big", "0");
      const line = el("div", "rp-kb2-line");
      line.append(this.kb2Count, el("span", "rp-label", "Notes written by the rounds"));
      this.kb2Segs = el("div", "rp-segs");
      this.segs = Array.from({ length: 40 }, () => { const s = el("div", "rp-seg"); this.kb2Segs.append(s); return s; });
      top.append(line, this.kb2Segs);
      this.cards = el("div", "rp-cards");
      p.body.append(top, el("div", "rp-rule"), this.cards);
      return p;
    }

    buildCover(closing) {
      const root = el("div", "rp-cover");
      const hero = el("img", "rp-hero");
      hero.alt = "";
      root.append(hero, el("div", "rp-veil"));
      const center = el("div", "rp-cover-center");
      const mark = el("div", "rp-cover-mark");
      mark.append(svgMark(FORGE_SVG, 82));
      const title = el("div", "rp-cover-title", "PINNForge");
      const sub = el("div", "rp-cover-sub", "Knowledge-centric agent system for the autonomous discovery of PINN solvers");
      const chips = el("div", "rp-cover-chips");
      const chipEls = ["Literature base", "Experience base", "Agent harness"].map((k) => {
        const c = el("div", "rp-cover-chip");
        c.append(el("i"), el("span", null, k));
        chips.append(c);
        return c;
      });
      center.append(mark, title, sub, chips);
      let result = null, logline = null;
      if (closing) {
        const d = this.d;
        result = el("div", "rp-cover-result");
        const factor = d.factor ? `${d.factor >= 10 ? d.factor.toFixed(0) : d.factor.toFixed(1)}×` : "—";
        result.append(el("span", "rp-strong", d.name || d.task), el("span", null, "error"),
          el("span", null, d.baseline != null ? d.baseline.toFixed(4) : "—"), el("span", "rp-faint", "→"),
          el("span", "rp-accent", sci(d.final)), el("span", "rp-strong", factor), el("span", "rp-faint", "lower error"));
        center.append(result);
        if (d.logline) center.append(logline = el("div", "rp-cover-logline", d.logline));
      }
      const logoBox = el("div", "rp-cover-logo");
      const logo = el("img");
      logo.alt = "COLALab — Computational Optimization for Learning & Adaptive Systems";
      logo.src = (window.__PF_ASSETS__ && window.__PF_ASSETS__.colalab) || "colalab.png";
      logoBox.append(logo);
      root.append(center, logoBox);
      return { root, hero, mark, title, sub, chipEls, result, logline, logoBox, heroSet: false };
    }

    // ── per block: what is new on the panels ──
    enterBlock(cur) {
      const d = this.d, b = cur.b;
      this.agent.setRight([b.id, fmtDuration(b.cost.duration_ms)]);
      this.headline.textContent = b.headline || b.id;
      const fork = b.fork && b.fork !== "baseline" ? `Forked from ${b.fork}` : b.fork === "baseline" ? "Forked from the standard PINN" : "";
      this.thrustText = b.thrust || (b.id === "b00" ? "Round 0: the standard PINN of the task, trained once — the anchor every round is measured against." : "");
      this.forkText = fork;
      this.tags.replaceChildren(...(b.tags || []).slice(0, 6).map((t) => el("span", "rp-tag", t)));
      this.quote.textContent = dropEnum((b.findings || [])[0] || "").slice(0, 170);
      const roster = (b.roster || []).slice(0, 9);
      this.rosterRows = roster.map((r) => {
        const row = el("div", `rp-cand ${r.st}`);
        const name = el("span", "rp-cand-id", r.label || r.id);
        if (r.label) name.title = r.id;
        row.append(el("i", "rp-dot"), name, el("span", "rp-cand-r", r.r == null ? (r.st === "failed" ? "failed" : "smoke") : sci(r.r)));
        return row;
      });
      this.rosterList.replaceChildren(...this.rosterRows);
      if (!roster.length) this.rosterList.append(el("div", "rp-faint", "no candidate scored"));
      const oc = Object.entries(b.outcomes || {}).sort((p, q) => ["winner", "improved", "regressed", "smoke_only", "failed"].indexOf(p[0]) - ["winner", "improved", "regressed", "smoke_only", "failed"].indexOf(q[0])).slice(0, 3);
      this.outcomes.replaceChildren(...oc.map(([k, v]) => {
        const c = el("div", `rp-outcome ${k}`);
        c.append(el("div", "rp-outcome-v", String(v)), el("div", "rp-label", sentence(k === "smoke_only" ? "smoke" : k.replace("_", " "))));
        return c;
      }));
      this.cost.setRight([b.cost.model || (b.id === "b00" ? "baseline" : "")]);
      this.citedRows = (b.cited || []).slice(0, 3).map((c) => {
        const r = el("div", "rp-cite");
        r.append(el("span", "rp-cite-id", c.id), el("span", "rp-cite-t", c.t), el("span", "rp-cite-y", String(c.y)));
        return r;
      });
      this.citedList.replaceChildren(...this.citedRows);
      if (!this.citedRows.length) this.citedList.append(el("div", "rp-faint", b.id === "b00" ? "no citation — control block" : "no paper cited in this block"));
      this.kb1.setRight([`${d.kb1.n} papers`, `${b.theory_cum} cited`]);
      this.showFields(cur.i);
      this.kb2Shown = -1;
    }

    // ── one frame ──
    render() {
      const T = this.T, tl = this.tl, d = this.d;
      const cur = cursor(clamp(T, tl.caseStart, tl.caseEnd), tl, d);
      if (cur.i !== this.lastI) { this.lastI = cur.i; this.enterBlock(cur); }
      const dash = MOTION.enter(T, tl.caseStart - 0.55, 0.75);
      this.dash.style.opacity = dash;
      for (const [p, lead] of this.panels) {
        const a = MOTION.enter(T, tl.caseStart - lead, 0.6);
        p.root.style.opacity = a;
        p.root.style.transform = `translateY(${(1 - a) * 14}px)`;
      }
      const kbA = MOTION.enter(T, tl.caseStart - 0.43, 0.6);
      this.kbBox.style.opacity = kbA;
      this.kbBox.style.transform = `translateY(${(1 - kbA) * 14}px)`;
      // header
      const live = cur.started ? cur.phase : "init";
      this.header.status.textContent = T < tl.caseStart ? "" : `${cur.b.id} / ${pad2(d.blocks.length - 1)} · ${sentence(live)}`;
      // equations
      this.eqRows.forEach((r, i) => { const a = MOTION.enter(T, tl.caseStart + 0.12 + i * 0.1, 0.5); r.style.opacity = a; r.style.transform = `translateX(${(1 - a) * -10}px)`; });
      this.eq.rule.style.opacity = MOTION.enter(T, tl.caseStart + 0.6, 0.4);
      this.eq.facts.style.opacity = MOTION.enter(T, tl.caseStart + 0.65, 0.5);
      // agent
      const glow = cur.phase === "thinking" || cur.phase === "retrieving" ? 0.5 + 0.5 * Math.sin(T * 7) : cur.phase === "executing" ? 0.5 : 0.25;
      this.mark.style.setProperty("--glow", glow.toFixed(3));
      this.ring.style.opacity = 0.15 + glow * 0.5;
      this.ring.style.transform = `scale(${1 + glow * 0.08})`;
      this.phaseEl.textContent = sentence(cur.phase);
      const think = this.thrustText || "";
      const n = Math.round(clamp((cur.ph - 0.02) / 0.26, 0, 1) * think.length);
      const typing = n < think.length;
      const nodes = [el("span", "rp-faint", "▸ ")];
      if (this.forkText) nodes.push(el("span", "rp-fork", this.forkText + " · "));
      nodes.push(document.createTextNode(think.slice(0, n)));
      if (typing) { const c = el("span", "rp-caret", "▌"); c.style.opacity = Math.sin(T * 12) > 0 ? 1 : 0.15; nodes.push(c); }
      this.thrust.replaceChildren(...nodes);
      [...this.tags.children].forEach((t, i) => { const a = clamp((cur.ph - 0.22 - i * 0.045) / 0.1, 0, 1); t.style.opacity = a; t.style.transform = `translateY(${(1 - a) * 5}px)`; });
      this.quote.style.opacity = clamp((cur.ph - 0.7) / 0.14, 0, 1);
      const nVis = Math.round(clamp((cur.ph - 0.2) / 0.5, 0, 1) * this.rosterRows.length);
      this.rosterRows.forEach((r, i) => { const on = i < nVis; r.style.opacity = on ? 1 : 0.13; r.style.transform = `translateX(${on ? 0 : 8}px)`; });
      // convergence and cost
      this.champ.textContent = sci(cur.score);
      this.drawConv(cur);
      const budget = Math.max(d.budget_h || 0, cur.gpu, 1e-6);
      this.gpuText.replaceChildren(document.createTextNode(cur.gpu.toFixed(2)), el("span", "rp-faint", ` / ${budget >= 1 ? budget.toFixed(1) : budget.toFixed(2)}h`));
      this.gpuFill.style.width = `${clamp(cur.gpu / budget, 0, 1) * 100}%`;
      this.tokText.textContent = kfmt(cur.tokens);
      this.bars.forEach((bar, i) => {
        const t = d.blocks[i].cost.tokens || 0;
        const grow = i < cur.i ? 1 : i === cur.i ? clamp(cur.ph / 0.8, 0, 1) : 0;
        bar.style.height = `${Math.max(6, (t / d.maxTok) * 100 * grow)}%`;
        bar.className = `rp-tok-bar${i === cur.i ? " now" : i < cur.i ? " past" : ""}`;
      });
      const c = cur.b.counts || {};
      [Math.round(cur.tools), c.cand, c.full, c.smoke].forEach((v, i) => { this.countEls[i].textContent = v == null ? "—" : String(v); });
      // knowledge base 1
      const heat = cur.ph > 0.42 ? cur.b.heatAfter : (cur.i > 0 ? d.blocks[cur.i - 1].heatAfter : {});
      const maxHeat = Math.max(2, ...Object.values(cur.b.heatAfter));
      const citedNow = new Set((cur.b.cited || []).map((x) => x.slot));
      const pulse = cur.ph < 0.42 ? Math.max(0, Math.sin((cur.ph / 0.42) * Math.PI)) : 0;
      const stops = HEAT[this.theme === "dark" ? "dark" : "light"];
      for (const e of d.kb1.entries) {
        const cell = this.cells[e.s], v = heat[e.s] || 0;
        const inten = v ? 0.28 + 0.72 * Math.min(1, v / maxHeat) : 0;
        cell.style.background = inten ? heatColor(inten, stops) : "";
        const now = citedNow.has(e.s);
        cell.classList.toggle("now", now);
        cell.style.transform = now ? `scale(${1 + pulse * 0.16})` : "";
        cell.style.boxShadow = now ? `0 0 ${4 + pulse * 12}px rgba(191,90,52,${0.2 + pulse * 0.5})` : "";
      }
      (this.citedRows || []).forEach((r, i) => { const a = clamp((cur.ph - 0.04 - i * 0.06) / 0.12, 0, 1); r.style.opacity = a; r.style.transform = `translateX(${(1 - a) * -8}px)`; });
      // knowledge base 2
      const revealed = d.practice.filter((p) => p.bi < cur.i || (p.bi === cur.i && cur.ph >= 0.8 + 0.19 * ((p.ord + 1) / p.n)));
      if (revealed.length !== this.kb2Shown) {
        this.kb2Shown = revealed.length;
        this.kb2Count.textContent = String(revealed.length);
        this.kb2.setRight([`${revealed.length} entries`]);
        const maxCum = Math.max(d.practice.length, 40);
        this.segs.forEach((s, i) => s.classList.toggle("on", i / 40 < revealed.length / maxCum));
        this.cards.replaceChildren(...revealed.slice(-6).reverse().map((p, i) => {
          const card = el("div", `rp-card${i === 0 ? " new" : ""}`);
          card.style.opacity = 1 - i * 0.14;
          const head = el("div", "rp-card-head");
          head.append(el("span", "rp-card-id", p.id), el("span", "rp-label", sentence(p.k)));
          card.append(head, el("div", "rp-card-t", dropEnum(p.t)));
          return card;
        }));
        if (!revealed.length) this.cards.append(el("div", "rp-faint", "nothing written yet"));
      }
      const fl = this.kb2.root.querySelector(".rp-flame svg");
      if (fl) {
        const f = Math.sin(T * 8.6) * 0.6 + Math.sin(T * 13.3 + 1.1) * 0.4, sway = Math.sin(T * 4.3) * 0.55 + Math.sin(T * 7.1 + 0.7) * 0.45;
        fl.style.transform = `rotate(${sway * 5.5}deg) skewX(${sway * -4.5}deg) scaleY(${1 + f * 0.11}) scaleX(${1 - f * 0.06})`;
      }
      // covers
      this.cover(this.opening, T, 0, 1 - MOTION.enter(T, tl.caseStart - 0.6, 0.55));
      this.cover(this.closing, T, tl.closing - 0.15, MOTION.enter(T, tl.closing - 0.55, 0.7));
      // transport
      this.clock.textContent = this.fmtTime(T);
      if (document.activeElement !== this.scrub) this.scrub.value = String(Math.round(T * 100));
    }

    cover(c, T, t0, opacity) {
      c.root.style.opacity = opacity;
      c.root.style.visibility = opacity <= 0.001 ? "hidden" : "visible";
      if (opacity <= 0.001) return;
      if (!c.heroSet) {
        c.heroSet = true;
        const tile = this.field && this.field.cols && this.field.cols[0] && this.field.cols[0].tiles[0];
        const last = this.fields && [...this.fields.order].reverse().find((f) => f != null);
        if (tile && last != null) {
          this.showFields(this.d.blocks.length - 1);
          const canvas = tile.layers[`f${last}`];
          if (canvas) c.hero.src = canvas.toDataURL();
          this.showFields(this.lastI < 0 ? 0 : this.lastI);
        }
      }
      const tl = T - t0;
      const a = MOTION.enter(tl, 0.15, 0.9), b = MOTION.enter(tl, 0.6, 0.9), cc = MOTION.enter(tl, 1.0, 0.9);
      c.hero.style.opacity = 0.42 * a;
      c.hero.style.transform = `translate(-50%,-50%) scale(${1.04 + 0.05 * clamp(tl / 3, 0, 1)})`;
      c.mark.style.opacity = a; c.mark.style.transform = `translateY(${(1 - a) * 10}px)`;
      c.title.style.opacity = a; c.title.style.transform = `scale(${0.98 + a * 0.02})`;
      c.sub.style.opacity = b;
      c.chipEls.forEach((ch, i) => { const e = MOTION.enter(tl, 1.05 + i * 0.13, 0.7); ch.style.opacity = e * cc; ch.style.transform = `translateY(${(1 - e) * 8}px)`; });
      if (c.result) { const r = MOTION.enter(tl, 1.6, 0.7); c.result.style.opacity = r; c.result.style.transform = `translateY(${(1 - r) * 8}px)`; }
      if (c.logline) { const r = MOTION.enter(tl, 2.1, 0.8); c.logline.style.opacity = r; c.logline.style.transform = `translateY(${(1 - r) * 8}px)`; }
      c.logoBox.style.opacity = MOTION.enter(tl, 0, 0.5);
    }

    // ── transport ──
    seek(t) { this.T = clamp(t, 0, this.tl.total); this.render(); }
    play() {
      if (this.playing) return;
      if (this.T >= this.tl.total - 0.01) this.T = 0;
      this.playing = true;
      this.playBtn.textContent = "❚❚";
      let last = performance.now();
      const step = (now) => {
        if (!this.playing || this.dead) return;
        this.T = Math.min(this.tl.total, this.T + (now - last) / 1000);
        last = now;
        this.render();
        if (this.T >= this.tl.total) { this.pause(); return; }
        this.raf = requestAnimationFrame(step);
      };
      this.raf = requestAnimationFrame(step);
    }
    pause() { this.playing = false; cancelAnimationFrame(this.raf); if (this.playBtn) this.playBtn.textContent = "▶"; }
    toggle() { if (this.playing) this.pause(); else this.play(); }
  }

  window.PFReplay = {
    async mount(host, runId) {
      const r = new Replay(host, runId);
      const wait = el("div", "rp-loading");
      wait.append(svgMark(FORGE_SVG, 30), el("span", null, "Loading run archive…"));
      host.replaceChildren(wait);
      try { await r.load(); } catch (e) {
        host.replaceChildren(el("div", "card empty", e && e.status === 404 ? "Not found." : "This run's replay could not be loaded."));
        console.error(e);
      }
      return r;
    },
    _texToHtml: texToHtml,  // for tests
  };
})();
