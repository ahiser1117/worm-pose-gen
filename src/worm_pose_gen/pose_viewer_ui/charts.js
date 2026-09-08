"use strict";

// Timeline charts under the stage (synced series over the source's rows),
// the jump-to buttons that walk events in those series, and playback.

function chartSpecs() {
  const run = state.run;
  const prior = run.prior;
  const specs = [
    { id: "iou", label: "IoU · coverage", height: 56, lines: [{ key: "iou", color: "#57d68d" }, { key: "tube_coverage", color: "#50beff", dash: [2, 3] }, { key: "iou_independent", color: "#9aa6b0", dash: [3, 3], opt: "independent" }], compare: "iou", hline: () => parseFloat($("#jump-iou").value), ymin: 0, ymax: 1 },
    { id: "length", label: "Body length px", height: 56, lines: [{ key: "body_length_px", color: "#57d68d" }, { key: "track_length_px", color: "#9aa6b0", dash: [6, 4] }], compare: "body_length_px", band: prior ? [prior.length_px * Math.exp(-2 * prior.log_length_sigma), prior.length_px * Math.exp(2 * prior.log_length_sigma)] : null },
    { id: "area", label: "Area px (mask, tube in view, raw)", height: 56, lines: [{ key: "worm_pixels", color: "#50beff" }, { key: "tube_area_visible_px", color: "#57d68d" }, { key: "raw_worm_pixels", color: "#9aa6b0", dash: [2, 3] }] },
    { id: "score", label: "Ambiguity score", height: 44, bars: "ambiguity_score", dots: { key: "score_independent", color: "#9aa6b0", opt: "independent" }, ymin: 0, ymax: 9 },
    { id: "extra", label: "", height: 50, lines: [{ key: null, color: "#e0b04d" }], compare: null },
    { id: "flags", label: "Flags", height: 9 * 10 + 2, raster: true },
    { id: "strip", label: "Class · source", height: 20, strip: true },
  ];
  return specs;
}

function buildCharts() {
  const container = $("#charts");
  container.innerHTML = "";
  state.charts = [];
  const extra = $("#extra-series");
  if (!extra.options.length) {
    for (const [key, label] of EXTRA_SERIES) { const o = document.createElement("option"); o.value = key; o.textContent = label; extra.appendChild(o); }
    extra.value = state.extra;
  }
  if (!state.run) return;
  for (const spec of chartSpecs()) {
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = spec.label;
    const node = document.createElement("canvas");
    node.dataset.height = spec.height;
    node.style.height = `${spec.height}px`;
    container.appendChild(label); container.appendChild(node);
    node.addEventListener("pointerdown", (event) => { state.timeline.dragging = true; node.setPointerCapture(event.pointerId); seekFromEvent(node, event); });
    node.addEventListener("pointermove", (event) => { if (state.timeline.dragging) seekFromEvent(node, event); });
    node.addEventListener("pointerup", () => { state.timeline.dragging = false; });
    node.addEventListener("wheel", (event) => { event.preventDefault(); zoomTimeline(node, event); }, { passive: false });
    node.addEventListener("dblclick", () => { state.timeline.start = 0; state.timeline.end = state.run.series.frame_index.length; drawCharts(); });
    state.charts.push({ spec, node, label });
  }
  drawCharts();
}

function timelineX(node, rowIndex) {
  const { start, end } = state.timeline;
  const w = node.getBoundingClientRect().width;
  return ((rowIndex + 0.5 - start) / (end - start)) * w;
}

function rowFromX(node, clientX) {
  const rect = node.getBoundingClientRect();
  const { start, end } = state.timeline;
  return Math.floor(start + ((clientX - rect.left) / rect.width) * (end - start));
}

function seekFromEvent(node, event) {
  const r = rowFromX(node, event.clientX);
  if (r !== state.row) showRow(r, { keepView: true });
}

function zoomTimeline(node, event) {
  const n = state.run.series.frame_index.length;
  const t = state.timeline;
  const pivot = rowFromX(node, event.clientX);
  const factor = event.deltaY > 0 ? 1.25 : 0.8;
  let span = Math.max(20, Math.min(n, Math.round((t.end - t.start) * factor)));
  let start = Math.round(pivot - (pivot - t.start) * (span / (t.end - t.start)));
  start = Math.max(0, Math.min(n - span, start));
  state.timeline.start = start; state.timeline.end = start + span;
  drawCharts();
}

function compareValues(key) {
  if (!state.compare || !$("#show-compare").checked || !state.compare.series || !state.compare.series[key]) return null;
  const series = state.run.series;
  const values = state.compare.series[key];
  return series.frame_index.map((f) => { const r = state.compareRows.get(f); return r === undefined ? null : values[r]; });
}

let chartsFrame = null;
function drawCharts() {
  if (chartsFrame) return;
  chartsFrame = requestAnimationFrame(() => { chartsFrame = null; drawChartsNow(); });
}

function drawChartsNow() {
  if (!state.run || !state.run.series) return;
  const series = state.run.series;
  const n = series.frame_index.length;
  if (!n) { $("#timeline-range").textContent = "no frames"; return; }
  const { start, end } = state.timeline;
  $("#timeline-range").textContent = `frames ${series.frame_index[start]}–${series.frame_index[Math.max(start, end - 1)]} (${end - start} of ${n} rows)`;
  const showIndependent = $("#show-independent").checked;
  const flagGroups = (state.info && state.info.flag_groups) || {};
  for (const { spec, node, label } of state.charts) {
    const { c, w, h } = chartBox(node);
    const X = (i) => ((i + 0.5 - start) / (end - start)) * w;
    const colw = Math.max(1, w / (end - start));
    // Propagation stretches shaded on every chart.
    for (const [a, b] of state.run.stretches || []) {
      if (b < start || a >= end) continue;
      c.fillStyle = "rgba(255,255,255,0.06)";
      c.fillRect(X(a) - colw / 2, 0, X(b) - X(a) + colw, h);
    }
    if (spec.raster) {
      const names = Object.keys(series.flags || {});
      const rowh = (h - 2) / Math.max(1, names.length);
      names.forEach((name, k) => {
        const group = Object.entries(flagGroups).find(([, list]) => list.includes(name));
        c.fillStyle = GROUP_COLORS[group ? group[0] : "failure"];
        const values = series.flags[name];
        for (let i = start; i < end; i++) if (values[i]) c.fillRect(X(i) - colw / 2, 1 + k * rowh, Math.max(1, colw), rowh - 1);
      });
      label.innerHTML = names.map((nm) => `<div style="height:${rowh}px;line-height:${rowh}px;font-size:9px;overflow:hidden">${nm}</div>`).join("");
    } else if (spec.strip) {
      const half = (h - 2) / 2;
      for (let i = start; i < end; i++) {
        c.fillStyle = KIND_COLORS[series.classification ? series.classification[i] : (series.fitted[i] ? "clean" : "unfitted")] || "#444";
        c.fillRect(X(i) - colw / 2, 1, Math.max(1, colw), half - 1);
        if (series.source) {
          const s = series.source[i];
          c.fillStyle = s === 1 ? "#ffaa3c" : s === 2 ? "#c080ff" : series.fitted[i] ? "#2a343c" : "#111";
          c.fillRect(X(i) - colw / 2, 1 + half, Math.max(1, colw), half - 1);
        }
      }
    } else {
      const lines = spec.lines ? spec.lines.map((l) => ({ ...l, key: l.key === null ? state.extra : l.key })).filter((l) => series[l.key]) : [];
      if (spec.id === "extra") label.textContent = (EXTRA_SERIES.find(([k]) => k === state.extra) || [state.extra, state.extra])[1];
      const values = [];
      for (const l of lines) if (!(l.opt === "independent" && !showIndependent)) for (let i = start; i < end; i++) { const v = series[l.key][i]; if (v !== null && Number.isFinite(v)) values.push(v); }
      const compare = spec.compare ? compareValues(spec.id === "extra" ? state.extra : spec.compare) : (spec.id === "extra" ? compareValues(state.extra) : null);
      if (compare) for (let i = start; i < end; i++) if (compare[i] !== null && Number.isFinite(compare[i])) values.push(compare[i]);
      if (spec.bars && series[spec.bars]) for (let i = start; i < end; i++) values.push(series[spec.bars][i]);
      if (spec.band) values.push(...spec.band);
      let ymin = spec.ymin !== undefined ? spec.ymin : Math.min(...values);
      let ymax = spec.ymax !== undefined ? spec.ymax : Math.max(...values);
      if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) { ymin = 0; ymax = 1; }
      if (ymax - ymin < 1e-9) { ymax = ymin + 1; }
      if (spec.ymin === undefined) { const m = (ymax - ymin) * 0.08; ymin -= m; ymax += m; }
      const Y = (v) => 3 + (1 - (v - ymin) / (ymax - ymin)) * (h - 6);
      c.fillStyle = "#9aa6b0"; c.font = "9px system-ui";
      c.fillText(fmt(ymax, 2), 2, 10); c.fillText(fmt(ymin, 2), 2, h - 3);
      if (spec.band) {
        c.fillStyle = "rgba(87,214,141,0.12)";
        c.fillRect(0, Y(spec.band[1]), w, Y(spec.band[0]) - Y(spec.band[1]));
      }
      if (spec.hline) {
        const yv = spec.hline();
        c.strokeStyle = "rgba(255,93,93,0.6)"; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(0, Y(yv)); c.lineTo(w, Y(yv)); c.stroke(); c.setLineDash([]);
      }
      if (spec.bars && series[spec.bars]) {
        const vals = series[spec.bars];
        for (let i = start; i < end; i++) {
          if (!vals[i]) continue;
          c.fillStyle = vals[i] >= 2 ? "#ff5d5d" : "#e0b04d";
          c.fillRect(X(i) - colw / 2, Y(vals[i]), Math.max(1, colw), Y(0) - Y(vals[i]));
        }
        if (spec.dots && series[spec.dots.key] && !(spec.dots.opt === "independent" && !showIndependent)) {
          c.fillStyle = spec.dots.color;
          for (let i = start; i < end; i++) { const v = series[spec.dots.key][i]; if (v) c.fillRect(X(i) - 1, Y(v) - 1, 2, 2); }
        }
      }
      const xs = [];
      for (let i = start; i < end; i++) xs.push(X(i));
      for (const l of lines) {
        if (l.opt === "independent" && !showIndependent) continue;
        polyline(c, xs, series[l.key].slice(start, end).map((v) => (v === null ? null : Y(v))), l.color, l.dash, 1.2);
      }
      if (compare) polyline(c, xs, compare.slice(start, end).map((v) => (v === null ? null : Y(v))), "#ffaa3c", [3, 3], 1);
    }
    // Cursor.
    if (state.row >= start && state.row < end) {
      c.strokeStyle = "rgba(255,255,255,0.85)"; c.lineWidth = 1;
      c.beginPath(); c.moveTo(X(state.row), 0); c.lineTo(X(state.row), h); c.stroke();
    }
  }
}

// ---------------------------------------------------------------- jumps and playback

function jump(kind, direction) {
  if (!state.run) return;
  const series = state.run.series;
  const n = series.frame_index.length;
  const nPoints = state.run.n_points || 100;
  const minScore = parseInt($("#jump-score").value, 10) || 2;
  const maxIou = parseFloat($("#jump-iou").value);
  const flags = series.flags || {};
  const test = {
    flag: (r) => series.fitted[r] && series.ambiguity_score && series.ambiguity_score[r] >= minScore,
    iou: (r) => series.fitted[r] && series.iou && series.iou[r] !== null && series.iou[r] < maxIou,
    jump: (r) => flags.pose_jump && flags.pose_jump[r],
    edge: (r) => series.fitted[r] && ((series.mask_on_border && series.mask_on_border[r]) || (series.points_in_fov && series.points_in_fov[r] < nPoints) || (flags.edge_inside && flags.edge_inside[r])),
  }[kind];
  if (kind === "stretch") {
    const stretches = state.run.stretches || [];
    if (!stretches.length) { setStatus("no propagation stretches in this run", "error"); return; }
    let target = null;
    if (direction > 0) target = stretches.find(([a]) => a > state.row);
    else { const before = stretches.filter(([a]) => a < state.row && !(a <= state.row && state.row <= stretches.find(([x]) => x === a)[1])); target = before[before.length - 1]; }
    if (!target) { setStatus("no more stretches in that direction", "error"); return; }
    showRow(target[0], { keepView: true });
    return;
  }
  // Leave the current run of matching frames first, so repeated presses walk events, not frames.
  let r = state.row;
  while (r >= 0 && r < n && test(r)) r += direction;
  while (r >= 0 && r < n && !test(r)) r += direction;
  if (r < 0 || r >= n) { setStatus("no more matching frames in that direction", "error"); return; }
  showRow(r, { keepView: true });
}

function togglePlay() {
  if (state.playing) { clearInterval(state.playing); state.playing = null; $("#play").textContent = "Play"; return; }
  if (!state.run) return;
  const period = 1000 / Math.max(1, Math.min(30, parseInt($("#fps").value, 10) || 10));
  // The cursor advances at the requested rate; frames whose full layers are
  // already cached show them, the others show the light tier and the mask
  // layers catch up when playback stops.
  state.playing = setInterval(() => {
    const n = state.run.series.frame_index.length;
    let next = state.row + 1;
    const stretch = state.frame && state.frame.stats && state.frame.stats.stretch;
    if ($("#loop-stretch").checked && stretch && next > stretch.rows[1]) next = stretch.rows[0];
    if (next >= n) { togglePlay(); return; }
    showRow(next, { keepView: true, immediate: true });
  }, period);
  $("#play").textContent = "Pause";
}
