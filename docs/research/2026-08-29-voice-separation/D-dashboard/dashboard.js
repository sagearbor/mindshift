(function () {
  'use strict';
  const DATA = JSON.parse(document.getElementById('viz-data').textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const PX_PER_S = 24;          // timeline zoom
  const MERGE_COSINE = 0.45;    // diarize_local.MAX_POOLED_COSINE
  const FEATURE_PANELS = [
    { key: 'f0', title: 'Pitch (F0)', feats: ['f0'] },
    { key: 'centroid', title: 'Spectral centroid (brightness)', feats: ['centroid'] },
    { key: 'tilt', title: 'Spectral tilt (dark ↔ bright)', feats: ['tilt'] },
    { key: 'energy', title: 'Energy (RMS)', feats: ['energy'] },
    { key: 'formants', title: 'Formants F1 / F2 / F3 (vocal-tract shape)', feats: ['f1', 'f2', 'f3'] },
  ];
  const SHAPES = ['circle', 'square', 'diamond', 'triangle', 'cross', 'ring', 'circle', 'square'];

  // ---------- tiny DOM helpers ----------
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') el.className = v;
      else if (k === 'html') el.innerHTML = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const kid of kids) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return el;
  }
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(NS, tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    for (const kid of kids) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return el;
  }
  const fmtT = t => t.toFixed(1) + ' s';
  const pct = v => v == null ? '–' : Math.round(v * 100) + '%';
  function fmtV(feat, v) {
    if (v == null) return '–';
    if (feat === 'tilt') return v.toFixed(1) + ' dB/kHz';
    if (feat === 'energy') return v.toFixed(1) + ' dB';
    return Math.round(v) + ' Hz';
  }
  const OVERLAP = 'overlap';
  const isOverlap = l => Array.isArray(l) || l === OVERLAP;
  const labelStr = l => Array.isArray(l) ? l.join(' + ') : l;
  const colorOf = (d, spk) => {
    if (isOverlap(spk)) return 'var(--muted)';
    const i = d.speakers.indexOf(spk); return i < 0 ? 'var(--hatch)' : `var(--s${i + 1})`;
  };
  const shapeOf = (d, spk) => { if (isOverlap(spk)) return 'cross'; const i = d.speakers.indexOf(spk); return i < 0 ? 'ring' : SHAPES[i % SHAPES.length]; };

  function niceTicks(lo, hi, n) {
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => span / st <= n + 1) || mag * 10;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }
  function quantile(sorted, q) {
    if (!sorted.length) return null;
    const p = (sorted.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
  }

  // shared hatch pattern for "unmapped" production clusters
  const defsSvg = s('svg', { width: 0, height: 0, style: 'position:absolute' },
    s('defs', null, s('pattern', { id: 'hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' },
      s('rect', { width: 6, height: 6, fill: 'var(--surface)' }),
      s('rect', { width: 2.5, height: 6, fill: 'var(--hatch)' }))));
  document.body.prepend(defsSvg);

  // ---------- tooltip ----------
  function tooltipFor(container) {
    const tip = h('div', { class: 'tip' });
    container.appendChild(tip);
    return {
      show(x, y, html, below) {
        tip.innerHTML = html;
        tip.classList.add('show');
        tip.style.left = x + 'px';
        tip.style.top = (below ? y + 14 : y - 10) + 'px';
        tip.style.transform = below ? 'translate(-50%, 0)' : 'translate(-50%, -100%)';
        // keep inside the scroll box horizontally
        const w = tip.offsetWidth, cw = container.scrollWidth;
        if (x - w / 2 < 2) tip.style.left = (w / 2 + 2) + 'px';
        else if (x + w / 2 > cw - 2) tip.style.left = (cw - w / 2 - 2) + 'px';
      },
      hide() { tip.classList.remove('show'); },
    };
  }

  // ---------- timeline chart scaffold: fixed left axis + scrolling plot ----------
  function syncScrolls(list) {
    let lock = false;
    for (const el of list) el.addEventListener('scroll', () => {
      if (lock) return; lock = true;
      for (const o of list) if (o !== el) o.scrollLeft = el.scrollLeft;
      lock = false;
    });
  }
  function timeRow(d, opts) {
    const H = opts.height, LEFT = opts.left == null ? 44 : opts.left;
    const W = Math.ceil(d.duration_s * PX_PER_S) + 12;
    const row = h('div', { class: 'chart-row', style: `display:grid;grid-template-columns:${LEFT}px 1fr;align-items:start` });
    const axis = s('svg', { width: LEFT, height: H });
    const scroll = h('div', { class: 'scroll' });
    const svg = s('svg', { width: W, height: H });
    scroll.appendChild(svg);
    row.append(axis, scroll);
    return { row, axis, scroll, svg, W, x: t => t * PX_PER_S };
  }
  function xAxis(svg, d, y, x) {
    svg.appendChild(s('line', { class: 'axis', x1: 0, x2: x(d.duration_s), y1: y, y2: y }));
    const step = d.duration_s > 60 ? 10 : 5;
    for (let t = 0; t <= d.duration_s; t += step) {
      svg.appendChild(s('line', { class: 'axis', x1: x(t), x2: x(t), y1: y, y2: y + 4 }));
      svg.appendChild(s('text', { x: x(t), y: y + 15, 'text-anchor': t === 0 ? 'start' : 'middle' }, t + 's'));
    }
  }


  // ---------- 0. PRIMARY: raw features over time, one shared axis ----------
  // Lines are the raw measurements, NOT coloured by any speaker label; the
  // ground-truth rubric is an optional band overlay behind them.
  const OVERLAY = [
    { key: 'f0', label: 'Pitch (F0)', src: 'frame', on: true },
    { key: 'energy', label: 'Energy (RMS)', src: 'frame', on: false },
    { key: 'centroid', label: 'Spectral centroid', src: 'frame', on: true },
    { key: 'tilt', label: 'Spectral tilt', src: 'frame', on: false },
    { key: 'f1', label: 'Formant F1', src: 'frame', on: false },
    { key: 'f2', label: 'Formant F2', src: 'frame', on: false },
    { key: 'pc1', label: 'Voiceprint PC1', src: 'embed', idx: 0, on: true },
    { key: 'pc2', label: 'Voiceprint PC2', src: 'embed', idx: 1, on: false },
  ];
  const fmtPC = v => v.toFixed(2);
  // Feature toggles persist across fixture switches (and reloads, best-effort).
  const STATE = { feat: {}, gt: true, fixture: null };
  for (const o of OVERLAY) STATE.feat[o.key] = o.on;
  try {
    const saved = JSON.parse(localStorage.getItem('voice-dashboard-state') || 'null');
    if (saved && saved.feat) { Object.assign(STATE.feat, saved.feat); STATE.gt = saved.gt !== false; STATE.fixture = saved.fixture || null; }
  } catch (e) { /* storage unavailable: defaults */ }
  function saveState() { try { localStorage.setItem('voice-dashboard-state', JSON.stringify(STATE)); } catch (e) { /* ignore */ } }
  function overlaySeries(d) {
    const speechT = new Set(d.frames.map(f => Math.round(f.t * 10)));
    return OVERLAY.map((o, i) => {
      let pts;
      if (o.src === 'frame') {
        pts = d.frames.filter(f => f[o.key] != null).map(f => ({ t: f.t, v: f[o.key] }));
      } else {
        pts = d.embed.points.filter(p => p.pca.length > o.idx && speechT.has(Math.round(p.t * 10)))
          .map(p => ({ t: p.t, v: p.pca[o.idx] }));
      }
      const sorted = pts.map(p => p.v).sort((a, b) => a - b);
      let lo = quantile(sorted, 0.01), hi = quantile(sorted, 0.99);
      if (o.key === 'f0' && lo != null) lo = Math.max(50, lo);
      if (lo == null || hi === lo) { lo = (lo || 0) - 1; hi = (hi || 0) + 1; }
      const gap = o.src === 'frame' ? d.hop_s * 1.5 : d.embed.hop_s * 1.5;
      const unit = o.src === 'frame' ? (d.features[o.key] || {}).unit : '';
      const fmt = o.src === 'frame' ? (v => fmtV(o.key, v)) : fmtPC;
      return { ...o, on: STATE.feat[o.key] !== false, color: `var(--s${i + 1})`, pts, lo, hi, gap, unit, fmt, available: pts.length > 1 };
    });
  }
  function overlayChart(d, scrolls) {
    const series = overlaySeries(d);
    const hasGT = d.gt && d.gt.length > 0;
    const H = 270, TOP = hasGT ? 26 : 12, BOT = 22, PH = H - TOP - BOT;
    const tr = timeRow(d, { height: H, left: 36 });
    scrolls.push(tr.scroll);
    const ny = u => TOP + (1 - Math.max(-0.04, Math.min(1.04, u))) * PH;
    const tip = tooltipFor(tr.scroll);
    // fixed axis: normalized 0..1
    for (const u of [0, 0.5, 1]) {
      tr.axis.appendChild(s('text', { x: 32, y: ny(u) + 4, 'text-anchor': 'end' }, u === 0.5 ? '½' : u));
      tr.svg.appendChild(s('line', { class: 'grid', x1: 0, x2: tr.W, y1: ny(u), y2: ny(u) }));
    }
    // ground-truth overlay: strip on top + pale bands behind the lines
    const gtGroup = s('g');
    if (hasGT) {
      for (const [st, en, l] of d.gt) {
        const c = colorOf(d, l);
        gtGroup.appendChild(s('rect', { x: tr.x(st), y: TOP, width: tr.x(en) - tr.x(st), height: PH, fill: c, opacity: 0.13 }));
        gtGroup.appendChild(s('rect', { x: tr.x(st), y: 4, width: Math.max(1, tr.x(en) - tr.x(st)), height: 12, rx: 2, fill: c, class: 'strip-seg' }));
        if (tr.x(en) - tr.x(st) > 34) gtGroup.appendChild(s('text', { x: tr.x(st) + 4, y: 13.5, style: 'fill:#fff;font-size:9.5px;font-weight:600' }, labelStr(l)));
      }
    }
    tr.svg.appendChild(gtGroup);
    // lines
    const clip = 'oclip-' + d.fixture;
    tr.svg.appendChild(s('defs', null, s('clipPath', { id: clip }, s('rect', { x: 0, y: TOP - 6, width: tr.W, height: PH + 12 }))));
    const groups = {};
    for (const sr of series) {
      const g = s('g', { 'clip-path': `url(#${clip})`, style: sr.on ? '' : 'display:none' });
      let path = [], prev = null;
      const norm = v => (v - sr.lo) / (sr.hi - sr.lo);
      const flush = () => {
        if (path.length >= 2) g.appendChild(s('path', { class: 'line thin', d: 'M' + path.join('L'), style: `stroke:${sr.color}` }));
        else if (path.length === 1) g.appendChild(s('circle', { cx: path[0].split(',')[0], cy: path[0].split(',')[1], r: 1.8, fill: sr.color }));
        path = [];
      };
      for (const p of sr.pts) {
        if (prev && p.t - prev.t > sr.gap) flush();
        path.push(`${tr.x(p.t).toFixed(1)},${ny(norm(p.v)).toFixed(1)}`);
        prev = p;
      }
      flush();
      tr.svg.appendChild(g);
      groups[sr.key] = g;
      sr.norm = norm;
    }
    xAxis(tr.svg, d, TOP + PH, tr.x);
    // hover: crosshair + one dot per visible series + tooltip in real units
    const cross = s('line', { class: 'crosshair', y1: TOP, y2: TOP + PH, style: 'display:none' });
    tr.svg.appendChild(cross);
    const dots = {};
    for (const sr of series) { dots[sr.key] = s('circle', { r: 4, class: 'dot', fill: sr.color, style: 'display:none' }); tr.svg.appendChild(dots[sr.key]); }
    const hit = s('rect', { class: 'hit', x: 0, y: 0, width: tr.W, height: H });
    tr.svg.appendChild(hit);
    const nearest = (pts, t, tol) => {
      let a = 0, b = pts.length - 1;
      if (!pts.length) return null;
      while (a < b) { const m = (a + b) >> 1; if (pts[m].t < t) a = m + 1; else b = m; }
      let p = pts[a]; if (a > 0 && Math.abs(pts[a - 1].t - t) < Math.abs(p.t - t)) p = pts[a - 1];
      return Math.abs(p.t - t) <= tol ? p : null;
    };
    const hideAll = () => { cross.style.display = 'none'; for (const k in dots) dots[k].style.display = 'none'; tip.hide(); };
    hit.addEventListener('pointermove', ev => {
      const r = tr.svg.getBoundingClientRect(), t = (ev.clientX - r.left) / PX_PER_S;
      const px = tr.x(t);
      const lines = [];
      const gl = hasGT ? d.gt.find(([st, en]) => st <= t && t < en) : null;
      lines.push(`<b>${fmtT(t)}</b>${gl ? ` · ${labelStr(gl[2])}` : (hasGT ? ' · (no rubric label)' : '')}`);
      let any = false, topY = TOP + PH;
      for (const sr of series) {
        if (!sr.on || !sr.available) { dots[sr.key].style.display = 'none'; continue; }
        const p = nearest(sr.pts, t, sr.gap);
        if (!p) { dots[sr.key].style.display = 'none'; continue; }
        any = true;
        const py = ny(sr.norm(p.v));
        topY = Math.min(topY, py);
        dots[sr.key].setAttribute('cx', tr.x(p.t)); dots[sr.key].setAttribute('cy', py); dots[sr.key].style.display = '';
        lines.push(`<span style="color:${sr.color}">●</span> ${sr.label}: ${sr.fmt(p.v)}`);
      }
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.style.display = '';
      if (any || hasGT) tip.show(px, Math.max(TOP + 10, topY), lines.join('<br>'), topY < TOP + 90); else tip.hide();
    });
    hit.addEventListener('pointerleave', hideAll);
    // controls
    const ctl = h('div', { class: 'ctl' });
    const boxes = {};
    for (const sr of series) {
      const cb = h('input', { type: 'checkbox' });
      cb.checked = sr.on; cb.disabled = !sr.available;
      boxes[sr.key] = cb;
      cb.addEventListener('change', () => { sr.on = cb.checked; STATE.feat[sr.key] = cb.checked; saveState(); groups[sr.key].style.display = sr.on ? '' : 'none'; hideAll(); });
      const sw = h('span', { class: 'sw', style: `background:${sr.color}` });
      ctl.appendChild(h('label', { class: 'chk' + (sr.available ? '' : ' off') }, cb, sw, sr.label, sr.unit ? h('span', { class: 'unit' }, ' ' + sr.unit) : null));
    }
    let gtBox = null;
    if (hasGT) {
      gtBox = h('input', { type: 'checkbox' }); gtBox.checked = STATE.gt;
      gtGroup.style.display = STATE.gt ? '' : 'none';
      gtBox.addEventListener('change', () => { STATE.gt = gtBox.checked; saveState(); gtGroup.style.display = gtBox.checked ? '' : 'none'; });
      ctl.appendChild(h('label', { class: 'chk gt' }, gtBox, h('span', { class: 'sw', style: 'background:linear-gradient(90deg,' + d.speakers.map(sp => colorOf(d, sp)).join(',') + ')' }), 'Ground-truth bands'));
    }
    // called when this fixture is shown again: adopt toggles changed on another fixture
    const sync = () => {
      for (const sr of series) {
        sr.on = STATE.feat[sr.key] !== false;
        boxes[sr.key].checked = sr.on;
        groups[sr.key].style.display = sr.on ? '' : 'none';
      }
      if (gtBox) { gtBox.checked = STATE.gt; gtGroup.style.display = STATE.gt ? '' : 'none'; }
      hideAll();
    };
    const wrap = h('div');
    wrap.sync = sync;
    wrap.append(ctl, tr.row,
      h('div', { class: 'note', html: 'Each line is a raw measurement rescaled to 0–1 (its 1st–99th percentile); hover or tap for real units. Nothing here is coloured by any predicted speaker — the pale bands and the top strip are the owner’s rubric only. Voiceprint PCs are the 1.5 s ECAPA windows projected on their first principal components (silent windows dropped).' }));
    return wrap;
  }

  // ---------- 1. speaker strips ----------
  function stripsPanel(d, scrolls) {
    const rows = [{ label: 'Ground truth', segs: d.gt, map: null, acc: null }];
    for (const p of d.prod) {
      rows.push({ label: 'Production · ' + p.name, segs: p.segments, map: p.score.mapping || {}, prod: p });
    }
    const RH = 22, TOP = 4, H = TOP + rows.length * RH + 22;
    const tr = timeRow(d, { height: H, left: 118 });
    if (scrolls) scrolls.push(tr.scroll);
    const tip = tooltipFor(tr.scroll);
    rows.forEach((r, i) => {
      const y = TOP + i * RH;
      const lab = s('text', { class: 'lbl', x: 114, y: y + 15, 'text-anchor': 'end' }, r.label.replace('Production · ', 'prod · '));
      tr.axis.appendChild(lab);
      if (r.prod) {
        const t = s('title', null, r.label);
        lab.appendChild(t);
      }
      for (const [st, en, l] of r.segs) {
        const gtLabel = r.map ? r.map[l] : l;
        const mapped = r.map ? gtLabel != null : true;
        const shown = labelStr(l) + (isOverlap(l) ? ' (overlap — either counts)' : '');
        const rect = s('rect', {
          class: 'strip-seg' + (mapped ? '' : ' strip-unmapped'), x: tr.x(st), y: y + 3,
          width: Math.max(1, tr.x(en) - tr.x(st)), height: RH - 6, rx: 3,
          fill: mapped ? colorOf(d, gtLabel) : 'url(#hatch)',
        });
        const html = r.map
          ? `<b>${l}</b> → ${mapped ? gtLabel : 'no GT speaker (extra cluster)'}<br>${fmtT(st)} – ${fmtT(en)}`
          : `<b>${shown}</b><br>${fmtT(st)} – ${fmtT(en)}`;
        rect.addEventListener('pointerenter', ev => tip.show(tr.x((st + en) / 2), y + 3, html, i === 0));
        rect.addEventListener('pointermove', ev => tip.show(tr.x((st + en) / 2), y + 3, html, i === 0));
        rect.addEventListener('pointerleave', () => tip.hide());
        tr.svg.appendChild(rect);
      }
    });
    xAxis(tr.svg, d, TOP + rows.length * RH + 2, tr.x);
    const body = h('div');
    body.appendChild(tr.row);
    const notes = d.prod.map(p => {
      const sc = p.score;
      const kTxt = p.returned_none ? 'returned <b>None</b> (heard one voice)' : `found <b>${p.k_pred}</b> of ${d.k_true} voices`;
      const rec = Object.entries(sc.per_gt_recall).map(([g, v]) => `${g} ${pct(v)}`).join(', ');
      return `<b>${p.name}</b>: ${kTxt}, <b>${pct(sc.frame_accuracy)}</b> of speech frames right (per speaker: ${rec}${sc.owner_purity != null ? `; owner purity ${pct(sc.owner_purity)}` : ''}). ${p.pooled_cosine != null ? `Worst pooled pair cosine ${p.pooled_cosine.toFixed(3)}.` : ''}`;
    });
    const ovNote = d.overlap_segments ? ' Solid grey in the ground-truth row = two people talking at once (either speaker is credited; those frames are left out of the per-speaker statistics).' : '';
    body.appendChild(h('div', { class: 'note', html: notes.join('<br>') + '<br><span style="color:var(--muted)">Hatched = a production cluster with no one-to-one ground-truth match.' + ovNote + ' Hover a bar for its label.</span>' }));
    return body;
  }

  // ---------- 2. feature line chart + distribution row ----------
  function featureChart(d, feat, ttl) {
    const info = d.features[feat];
    const vals = d.frames.map(f => f[feat]).filter(v => v != null).sort((a, b) => a - b);
    if (!vals.length) return h('div', { class: 'note' }, 'no ' + info.label + ' values');
    let lo = quantile(vals, 0.01), hi = quantile(vals, 0.99);
    if (feat === 'f0') lo = Math.max(50, lo);
    const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
    const H = 190, TOP = 30, BOT = 22, PH = H - TOP - BOT;
    const tr = timeRow(d, { height: H });
    const y = v => TOP + (1 - (v - lo) / (hi - lo)) * PH;
    const tip = tooltipFor(tr.scroll);
    // axis (fixed)
    for (const tv of niceTicks(lo, hi, 4)) {
      tr.axis.appendChild(s('text', { x: 40, y: y(tv) + 4, 'text-anchor': 'end' }, feat === 'tilt' ? tv.toFixed(1) : Math.round(tv)));
      tr.svg.appendChild(s('line', { class: 'grid', x1: 0, x2: tr.W, y1: y(tv), y2: y(tv) }));
    }
    tr.axis.appendChild(s('text', { x: 40, y: H - 8, 'text-anchor': 'end' }, info.unit));
    // GT speech shading under the lines (very light) so gaps read as gaps
    for (const [st, en, l] of d.gt) {
      tr.svg.appendChild(s('rect', { x: tr.x(st), y: TOP, width: tr.x(en) - tr.x(st), height: PH, fill: colorOf(d, l), opacity: 0.06 }));
    }
    // lines: break on speaker change, silence gap, or missing value
    const clip = 'clip-' + feat + '-' + d.fixture;
    tr.svg.appendChild(s('defs', null, s('clipPath', { id: clip }, s('rect', { x: 0, y: TOP - 1, width: tr.W, height: PH + 2 }))));
    const g = s('g', { 'clip-path': `url(#${clip})` });
    let path = [], prev = null;
    const flush = () => {
      if (path.length >= 2) g.appendChild(s('path', { class: 'line', d: 'M' + path.join('L'), style: `stroke:${colorOf(d, prev.spk)}` }));
      else if (path.length === 1) g.appendChild(s('circle', { cx: path[0].split(',')[0], cy: path[0].split(',')[1], r: 2, fill: colorOf(d, prev.spk) }));
      path = [];
    };
    const idx = []; // frames with a value, for hover
    for (const f of d.frames) {
      const v = f[feat];
      if (v == null) { flush(); prev = f; continue; }
      if (prev && (prev.spk !== f.spk || f.t - prev.t > d.hop_s * 1.5)) flush();
      path.push(`${tr.x(f.t).toFixed(1)},${y(v).toFixed(1)}`);
      idx.push(f); prev = f;
    }
    flush();
    tr.svg.appendChild(g);
    xAxis(tr.svg, d, TOP + PH, tr.x);
    // hover crosshair
    const cross = s('line', { class: 'crosshair', y1: TOP, y2: TOP + PH, style: 'display:none' });
    const dot = s('circle', { r: 4.5, class: 'dot', style: 'display:none' });
    tr.svg.append(cross, dot);
    const hit = s('rect', { class: 'hit', x: 0, y: 0, width: tr.W, height: H });
    tr.svg.appendChild(hit);
    hit.addEventListener('pointermove', ev => {
      const r = tr.svg.getBoundingClientRect(), t = (ev.clientX - r.left) / PX_PER_S;
      let a = 0, b = idx.length - 1;
      while (a < b) { const m = (a + b) >> 1; if (idx[m].t < t) a = m + 1; else b = m; }
      let f = idx[a]; if (a > 0 && Math.abs(idx[a - 1].t - t) < Math.abs(f.t - t)) f = idx[a - 1];
      if (!f || Math.abs(f.t - t) > 0.5) { cross.style.display = dot.style.display = 'none'; tip.hide(); return; }
      const px = tr.x(f.t), py = y(f[feat]);
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.style.display = '';
      dot.setAttribute('cx', px); dot.setAttribute('cy', py); dot.setAttribute('fill', colorOf(d, f.spk)); dot.style.display = '';
      tip.show(px, py, `<b>${f.spk === OVERLAP ? 'overlap (two voices)' : (f.spk || 'unlabelled')}</b> · ${fmtT(f.t)}<br>${fmtV(feat, f[feat])}`, py < 48);
    });
    hit.addEventListener('pointerleave', () => { cross.style.display = dot.style.display = 'none'; tip.hide(); });
    return tr.row;
  }

  function distRow(d, feat, width) {
    const sep = d.sep[feat], per = sep.per_speaker;
    const spks = d.speakers.filter(sp => per[sp]);
    if (!spks.length) return h('div');
    const LEFT = 78, RIGHT = 12, RH = 20, TOP = 6;
    const W = Math.max(280, width), H = TOP + spks.length * RH + 20;
    let lo = Math.min(...spks.map(sp => per[sp].p10)), hi = Math.max(...spks.map(sp => per[sp].p90));
    const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
    const x = v => LEFT + (v - lo) / (hi - lo) * (W - LEFT - RIGHT);
    const svg = s('svg', { width: W, height: H });
    for (const tv of niceTicks(lo, hi, 5)) {
      svg.appendChild(s('line', { class: 'grid', x1: x(tv), x2: x(tv), y1: TOP, y2: TOP + spks.length * RH }));
      svg.appendChild(s('text', { x: x(tv), y: H - 5, 'text-anchor': 'middle' }, feat === 'tilt' ? tv.toFixed(1) : Math.round(tv)));
    }
    spks.forEach((sp, i) => {
      const p = per[sp], cy = TOP + i * RH + RH / 2, c = colorOf(d, sp);
      svg.appendChild(s('text', { class: 'lbl', x: LEFT - 6, y: cy + 4, 'text-anchor': 'end' }, sp));
      svg.appendChild(s('line', { class: 'whisker whisker-thin', x1: x(p.p10), x2: x(p.p90), y1: cy, y2: cy, style: `stroke:${c}` }));
      svg.appendChild(s('line', { class: 'whisker', x1: x(p.q1), x2: x(p.q3), y1: cy, y2: cy, style: `stroke:${c}` }));
      svg.appendChild(s('circle', { class: 'dot', cx: x(p.median), cy, r: 5, fill: c }));
    });
    return svg;
  }

  function distTable(d, feat) {
    const per = d.sep[feat].per_speaker, spks = d.speakers.filter(sp => per[sp]);
    const tbl = h('table', { class: 'dist' });
    tbl.appendChild(h('tr', null, h('th', null, 'speaker'), h('th', null, 'median'), h('th', null, 'Q1'), h('th', null, 'Q3'), h('th', null, 'frames')));
    for (const sp of spks) {
      const p = per[sp];
      tbl.appendChild(h('tr', null, h('td', null, sp), h('td', null, fmtV(feat, p.median)), h('td', null, fmtV(feat, p.q1)), h('td', null, fmtV(feat, p.q3)), h('td', null, p.n)));
    }
    return tbl;
  }

  function featurePanel(d, spec, width) {
    const primary = d.sep[spec.feats[0]];
    const best = spec.feats.map(f => d.sep[f]).filter(x => x.ratio != null).sort((a, b) => b.ratio - a.ratio)[0] || primary;
    const bestKey = spec.feats.find(f => d.sep[f] === best) || spec.feats[0];
    const meta = best.ratio == null ? 'not enough voiced frames'
      : `${spec.feats.length > 1 ? d.features[bestKey].label + ' ' : ''}separability <b>${best.ratio.toFixed(2)}</b> · alone labels <b>${pct(best.accuracy)}</b> of frames (chance ${pct(best.chance)})`;
    const det = h('details', { class: 'panel' },
      h('summary', null, h('span', { class: 'ptitle' }, spec.title), h('span', { class: 'pmeta', html: meta })));
    const body = h('div', { class: 'panel-body' });
    det.appendChild(body);
    let built = false;
    det.addEventListener('toggle', () => {
      if (!det.open || built) return; built = true;
      for (const f of spec.feats) {
        if (spec.feats.length > 1) body.appendChild(h('div', { class: 'subhead' }, d.features[f].label));
        body.appendChild(featureChart(d, f, d.features[f].label));
        const sp = d.sep[f];
        body.appendChild(h('div', { class: 'subhead' }, 'Per-speaker median · IQR (thick) · 10–90% (thin)'));
        body.appendChild(h('div', { class: 'scroll' }, distRow(d, f, Math.min(640, body.clientWidth))));
        if (sp.ratio != null) body.appendChild(h('div', { class: 'note', html: `Spread of medians ÷ within-speaker spread = <b>${sp.ratio.toFixed(2)}</b>; closest two speakers are ${sp.min_gap_sigma.toFixed(2)} σ apart; nearest-median labelling alone gets <b>${pct(sp.accuracy)}</b> (chance ${pct(sp.chance)}).` }));
        body.appendChild(h('div', { class: 'scroll' }, distTable(d, f)));
      }
    });
    return det;
  }

  // ---------- 3. embedding scatter + pooled cosine heatmap ----------
  function marker(shape, cx, cy, r, attrs) {
    const a = Object.assign({}, attrs);
    switch (shape) {
      case 'square': return s('rect', Object.assign(a, { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, rx: 1 }));
      case 'diamond': return s('polygon', Object.assign(a, { points: `${cx},${cy - r * 1.2} ${cx + r * 1.2},${cy} ${cx},${cy + r * 1.2} ${cx - r * 1.2},${cy}` }));
      case 'triangle': return s('polygon', Object.assign(a, { points: `${cx},${cy - r * 1.2} ${cx + r * 1.15},${cy + r * 0.9} ${cx - r * 1.15},${cy + r * 0.9}` }));
      case 'cross': return s('path', Object.assign(a, { d: `M${cx - r},${cy - r}L${cx + r},${cy + r}M${cx - r},${cy + r}L${cx + r},${cy - r}`, 'stroke-width': 3, stroke: a.fill, fill: 'none' }));
      case 'ring': return s('circle', Object.assign(a, { cx, cy, r, fill: 'none', stroke: a.fill, 'stroke-width': 2.5 }));
      default: return s('circle', Object.assign(a, { cx, cy, r }));
    }
  }
  function scatter(d, proj, size) {
    const pts = d.embed.points, S = size, M = 16;
    const xs = pts.map(p => p[proj][0]), ys = pts.map(p => p[proj][1]);
    const xlo = Math.min(...xs), xhi = Math.max(...xs), ylo = Math.min(...ys), yhi = Math.max(...ys);
    const x = v => M + (v - xlo) / ((xhi - xlo) || 1) * (S - 2 * M), y = v => S - M - (v - ylo) / ((yhi - ylo) || 1) * (S - 2 * M);
    const wrap = h('div', { class: 'chart-wrap', style: `width:${S}px;max-width:100%` });
    const svg = s('svg', { width: S, height: S, style: 'max-width:100%;height:auto', viewBox: `0 0 ${S} ${S}` });
    svg.appendChild(s('rect', { x: 0.5, y: 0.5, width: S - 1, height: S - 1, fill: 'none', class: 'axis' }));
    const ax = proj === 'pca' ? `PC1 (${pct(d.embed.pca_explained[0])})` : 't-SNE 1';
    const ay = proj === 'pca' ? `PC2 (${pct(d.embed.pca_explained[1])})` : 't-SNE 2';
    svg.appendChild(s('text', { x: S - 6, y: S - 5, 'text-anchor': 'end' }, ax));
    svg.appendChild(s('text', { x: 5, y: 12 }, ay));
    const tip = tooltipFor(wrap);
    // unlabelled first (behind), then labelled
    const order = pts.slice().sort((a, b) => (a.spk == null ? -1 : 1) - (b.spk == null ? -1 : 1));
    for (const p of order) {
      const cx = x(p[proj][0]), cy = y(p[proj][1]);
      const m = marker(shapeOf(d, p.spk), cx, cy, 4.5, { class: 'pt' + (p.spk == null ? ' dim' : ''), fill: colorOf(d, p.spk) });
      svg.appendChild(m);
      const hit = s('circle', { class: 'hit', cx, cy, r: 9 });
      const html = `<b>${p.spk === OVERLAP ? 'overlap (two voices)' : (p.spk || 'outside GT speech')}</b> · window at ${fmtT(p.t)}`;
      hit.addEventListener('pointerenter', () => tip.show(cx, cy - 6, html, cy < 40));
      hit.addEventListener('pointerleave', () => tip.hide());
      svg.appendChild(hit);
    }
    wrap.appendChild(svg);
    return wrap;
  }
  function heatmap(d, width) {
    const k = d.speakers.length, LEFT = 72, TOPM = 20;
    const cell = Math.max(30, Math.min(58, Math.floor((Math.min(width, 420) - LEFT - 10) / k)));
    const W = LEFT + k * cell + 8, H = TOPM + k * cell + 8;
    const svg = s('svg', { width: W, height: H });
    const short = sp => sp.length > 9 ? sp.slice(0, 8) + '…' : sp;
    d.speakers.forEach((sp, i) => {
      svg.appendChild(s('text', { class: 'lbl', x: LEFT + i * cell + cell / 2, y: TOPM - 6, 'text-anchor': 'middle', style: `fill:${colorOf(d, sp)}` }, short(sp)));
      svg.appendChild(s('text', { class: 'lbl', x: LEFT - 6, y: TOPM + i * cell + cell / 2 + 4, 'text-anchor': 'end', style: `fill:${colorOf(d, sp)}` }, short(sp)));
    });
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      const v = d.pooled.cosine[i][j];
      const step = Math.min(7, Math.max(1, 1 + Math.round(Math.max(0, v) * 6)));
      const g = s('g');
      g.appendChild(s('rect', { x: LEFT + j * cell + 1, y: TOPM + i * cell + 1, width: cell - 2, height: cell - 2, rx: 4, fill: `var(--seq${step})` }));
      g.appendChild(s('text', { class: 'cell-text', x: LEFT + j * cell + cell / 2, y: TOPM + i * cell + cell / 2 + 4, 'text-anchor': 'middle',
        style: `fill:${step >= 5 ? 'var(--surface)' : 'var(--ink)'}` }, i === j ? '1' : v.toFixed(2)));
      if (i !== j && v > MERGE_COSINE) g.appendChild(s('rect', { x: LEFT + j * cell + 1, y: TOPM + i * cell + 1, width: cell - 2, height: cell - 2, rx: 4, fill: 'none', stroke: 'var(--critical)', 'stroke-width': 2 }));
      svg.appendChild(g);
    }
    return svg;
  }
  function embeddingPanel(d, width) {
    const e = d.embed, p = d.pooled;
    const worst = p.pairs.slice().sort((a, b) => b[2] - a[2])[0];
    const meta = `nearest-voiceprint window accuracy <b>${pct(p.nearest_centroid_window_accuracy)}</b> · silhouette <b>${e.silhouette_cosine == null ? '–' : e.silhouette_cosine.toFixed(2)}</b> · closest pair cosine <b>${worst ? worst[2].toFixed(2) : '–'}</b>`;
    const det = h('details', { class: 'panel' },
      h('summary', null, h('span', { class: 'ptitle' }, 'Speaker-ID model (ECAPA) embeddings'), h('span', { class: 'pmeta', html: meta })));
    const body = h('div', { class: 'panel-body' });
    det.appendChild(body);
    let built = false;
    det.addEventListener('toggle', () => {
      if (!det.open || built) return; built = true;
      const row = h('div', { class: 'row2' });
      const left = h('div'), right = h('div');
      const head = h('div', { class: 'subhead' }, `${e.window_s} s windows every ${e.hop_s} s, projected to 2-D`);
      const tog = h('span', { class: 'toggle' });
      const holder = h('div');
      const size = Math.min(380, Math.max(260, Math.floor(Math.min(width, 560) * 0.9)));
      let proj = 'pca';
      const btns = { pca: h('button', { class: 'on' }, 'PCA'), tsne: h('button', null, 't-SNE') };
      const redraw = () => { holder.innerHTML = ''; holder.appendChild(scatter(d, proj, size)); btns.pca.className = proj === 'pca' ? 'on' : ''; btns.tsne.className = proj === 'tsne' ? 'on' : ''; };
      btns.pca.addEventListener('click', () => { proj = 'pca'; redraw(); });
      btns.tsne.addEventListener('click', () => { proj = 'tsne'; redraw(); });
      tog.append(btns.pca, btns.tsne); head.appendChild(tog);
      left.append(head, holder); redraw();
      left.appendChild(h('div', { class: 'note', html: `Each mark is one window; colour and shape = who is really talking at its centre (grey rings = outside labelled speech${d.overlap_segments ? ', grey crosses = overlapping voices' : ''}). PCA keeps ${pct(e.pca_explained[0] + e.pca_explained[1])} of the variance of the 192-d vectors, so clusters that touch here may still be far apart in full dimension — the heatmap is the honest distance.` }));
      right.appendChild(h('div', { class: 'subhead' }, 'Cosine similarity of pooled per-speaker voiceprints'));
      right.appendChild(h('div', { class: 'scroll' }, heatmap(d, right.clientWidth || width)));
      const risky = p.pairs.filter(x => x[2] > MERGE_COSINE).map(x => `${x[0]}–${x[1]} (${x[2].toFixed(2)})`);
      right.appendChild(h('div', { class: 'note', html: `Same voice pooled twice ≈ 0.73, different voices ≈ 0.19 (speaker_id calibration). Production refuses a split whose centroids exceed <b>${MERGE_COSINE}</b>${risky.length ? ` — outlined in red: <b>${risky.join(', ')}</b>` : ' — no pair here comes close, so every voice is distinct to the model'}. Assigning each window to its nearest pooled voiceprint labels <b>${pct(p.nearest_centroid_window_accuracy)}</b> of windows correctly.` }));
      row.append(left, right);
      body.appendChild(row);
    });
    return det;
  }

  // ---------- 4. "what separates these voices" ----------
  function separabilityPanel(d) {
    const ranked = Object.entries(d.sep).filter(([, v]) => v.ratio != null).sort((a, b) => b[1].ratio - a[1].ratio);
    const det = h('details', { class: 'panel', open: '' },
      h('summary', null, h('span', { class: 'ptitle' }, 'What separates these voices'),
        h('span', { class: 'pmeta', html: ranked.length ? `best single feature: <b>${d.features[ranked[0][0]].label}</b> (${ranked[0][1].ratio.toFixed(2)})` : '' })));
    const body = h('div', { class: 'panel-body' });
    const ol = h('ol', { class: 'rank' });
    const maxR = Math.max(1, ...ranked.map(([, v]) => v.ratio));
    for (const [f, v] of ranked) {
      const meds = d.speakers.filter(sp => v.per_speaker[sp]).map(sp => v.per_speaker[sp].median);
      const iqr = d.speakers.filter(sp => v.per_speaker[sp]).map(sp => v.per_speaker[sp].q3 - v.per_speaker[sp].q1);
      const meanIqr = iqr.reduce((a, b) => a + b, 0) / iqr.length;
      const verdict = v.ratio >= 2 ? 'clearly separates' : v.ratio >= 1 ? 'partly separates' : 'overlaps';
      ol.appendChild(h('li', { html: `<span class="bar" style="width:${Math.round(v.ratio / maxR * 90)}px"></span><b>${d.features[f].label}</b> <span class="r">ratio ${v.ratio.toFixed(2)}</span> — ${verdict}: medians ${fmtV(f, Math.min(...meds))} to ${fmtV(f, Math.max(...meds))}, typical within-speaker IQR ${fmtV(f, meanIqr)}; alone labels ${pct(v.accuracy)} of frames (chance ${pct(v.chance)}).` }));
    }
    body.appendChild(ol);
    const p = d.pooled, worst = p.pairs.slice().sort((a, b) => b[2] - a[2])[0];
    const prod = d.prod[0];
    body.appendChild(h('div', { class: 'note', html:
      `<b>Speaker-ID model:</b> nearest-voiceprint window accuracy ${pct(p.nearest_centroid_window_accuracy)}; the two most alike voices to the model are ${worst ? `${worst[0]} and ${worst[1]} at cosine ${worst[2].toFixed(2)}` : '–'} (production merges above ${MERGE_COSINE}). ` +
      `<b>Production today</b> (GT turns as input): ${prod.returned_none ? 'heard one voice' : `${prod.k_pred} of ${d.k_true} voices`}, ${pct(prod.score.frame_accuracy)} frames right.` }));
    det.appendChild(body);
    return det;
  }

  // ---------- fixture section + sticky switcher ----------
  function legend(d) {
    const el = h('div', { class: 'legend' });
    d.speakers.forEach((sp, i) => {
      const sw = s('svg', { width: 14, height: 14, style: 'vertical-align:-2px;margin-right:5px' });
      sw.appendChild(marker(shapeOf(d, sp), 7, 7, 4.5, { fill: colorOf(d, sp) }));
      el.appendChild(h('span', null, sw, sp, sp === d.owner ? h('span', { class: 'owner' }, '(owner)') : null));
    });
    if (d.overlap_segments) {
      const sw = s('svg', { width: 14, height: 14, style: 'vertical-align:-2px;margin-right:5px' });
      sw.appendChild(marker('cross', 7, 7, 4.5, { fill: 'var(--muted)' }));
      el.appendChild(h('span', null, sw, 'overlap (two at once)'));
    }
    return el;
  }
  function fixtureSection(d) {
    const prod = d.prod[0], sc = prod.score;
    const ok = !prod.returned_none && prod.k_pred === d.k_true && sc.frame_accuracy >= 0.85;
    const cls = prod.returned_none ? 'bad' : ok ? 'ok' : 'warn';
    const sec = h('section', { class: 'fixture', id: 'fx-' + d.fixture, style: 'display:none' },
      h('div', { class: 'fx-head' },
        h('span', { class: 'name' }, d.fixture),
        h('span', { class: 'sub' }, d.title),
        h('span', { class: 'stat', html: `<b>${d.k_true}</b> voices · ${d.duration_s.toFixed(0)} s` }),
        h('span', { class: 'pill ' + cls }, prod.returned_none ? 'production: 1 voice' : `production: ${prod.k_pred} of ${d.k_true} · ${pct(sc.frame_accuracy)} right`)));
    const body = h('div', { class: 'fixture-body' });
    sec.appendChild(body);
    let built = false, dirty = false, overlay = null;
    const build = (openIdx) => {
      body.innerHTML = '';
      if (d.private) body.appendChild(h('p', { class: 'private-note' }, 'Private recording: only derived numbers are embedded here, no audio.'));
      body.appendChild(legend(d));
      const scrolls = [];
      overlay = overlayChart(d, scrolls);
      body.appendChild(overlay);
      const w = body.clientWidth || 360;
      const prodDet = h('details', { class: 'panel' },
        h('summary', null, h('span', { class: 'ptitle' }, 'Production diarizer today (for comparison)'),
          h('span', { class: 'pmeta', html: prod.returned_none ? 'heard <b>one</b> voice' : `<b>${prod.k_pred}</b> of ${d.k_true} voices · <b>${pct(sc.frame_accuracy)}</b> frames right` })));
      let prodBuilt = false;
      prodDet.addEventListener('toggle', () => {
        if (!prodDet.open || prodBuilt) return; prodBuilt = true;
        prodDet.appendChild(h('div', { class: 'panel-body' }, stripsPanel(d, scrolls)));
        syncScrolls(scrolls);
      });
      body.appendChild(prodDet);
      const perDet = h('details', { class: 'panel' },
        h('summary', null, h('span', { class: 'ptitle' }, 'Per-speaker breakdown (lines coloured by ground truth)'),
          h('span', { class: 'pmeta' }, 'one section per feature · median ± IQR per speaker')));
      const perBody = h('div', { class: 'panel-body' });
      for (const spec of FEATURE_PANELS) perBody.appendChild(featurePanel(d, spec, w));
      perDet.appendChild(perBody);
      body.appendChild(perDet);
      body.appendChild(embeddingPanel(d, w));
      body.appendChild(separabilityPanel(d));
      syncScrolls(scrolls);
      if (openIdx) body.querySelectorAll('details.panel').forEach((p, i) => { p.open = openIdx.has(i); });
      built = true; dirty = false;
    };
    sec.show = () => {
      sec.style.display = '';
      if (!built || dirty) build(); else if (overlay && overlay.sync) overlay.sync();
    };
    sec.hide = () => { sec.style.display = 'none'; };
    // re-layout on a real width change (rotation, window resize), keeping open panels open
    REBUILD.push(() => {
      if (!built) return;
      if (sec.style.display === 'none') { dirty = true; return; }
      const openIdx = new Set([...body.querySelectorAll('details.panel')].map((p, i) => p.open ? i : -1).filter(i => i >= 0));
      build(openIdx);
    });
    return sec;
  }
  const REBUILD = [];
  let lastW = window.innerWidth, resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (window.innerWidth !== lastW) { lastW = window.innerWidth; REBUILD.forEach(f => f()); } }, 250);
  });

  const app = document.getElementById('app');
  const nav = document.getElementById('switcher');
  const sections = {}, buttons = {};
  for (const d of DATA) {
    sections[d.fixture] = fixtureSection(d);
    app.appendChild(sections[d.fixture]);
    const b = h('button', { type: 'button', 'data-fx': d.fixture }, d.fixture);
    b.addEventListener('click', () => select(d.fixture));
    buttons[d.fixture] = b;
    nav.appendChild(b);
  }
  function select(name) {
    if (!sections[name]) return;
    for (const k in sections) { if (k !== name) sections[k].hide(); buttons[k].classList.toggle('on', k === name); }
    sections[name].show();
    STATE.fixture = name; saveState();
    if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
    window.scrollTo({ top: 0 });
  }
  const fromHash = location.hash.replace('#', '');
  const initial = sections[fromHash] ? fromHash : (sections[STATE.fixture] ? STATE.fixture : (sections.maggiano3 ? 'maggiano3' : DATA[0] && DATA[0].fixture));
  if (initial) select(initial);
  window.addEventListener('hashchange', () => { const n = location.hash.replace('#', ''); if (sections[n]) select(n); });
})();
