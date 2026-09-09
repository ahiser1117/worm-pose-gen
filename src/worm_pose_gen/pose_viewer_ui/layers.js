"use strict";

// Frame rendering: composite the pixel layers of the current frame into an
// overlay, draw the vector layers on the stage canvas, and fill the details
// panel (statistics, flags, provenance, width and curvature charts).

const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
const overlayCanvas = document.createElement("canvas");
const overlayCtx = overlayCanvas.getContext("2d");

// ---------------------------------------------------------------- compositing

function layer(id) { return LAYERS.find((l) => l.id === id); }

function buildOverlay() {
  const d = state.decoded;
  if (!d || !d.width) return;
  const W = d.width, H = d.height, N = W * H;
  overlayCanvas.width = W; overlayCanvas.height = H;
  const out = overlayCtx.createImageData(W, H);
  const px = out.data;
  const on = (id) => { const l = layer(id); return l.on && l.alpha > 0 ? l : null; };
  const prob = on("probability"), raw = on("mask_raw"), adds = on("fill_adds"), drops = on("largest_drops");
  const residual = on("residual"), tube = on("tube"), tubeFill = on("tube_fill"), outline = on("final_outline"), indep = on("independent");
  const P = d.probability, MR = d.mask_raw, MF = d.mask_filled, ML = d.mask_largest, MX = d.mask_final, T = d.tube, TI = d.tube_independent;
  const blend = (i, color, alpha) => {
    const o = i * 4, a = px[o + 3] / 255;
    const na = alpha + a * (1 - alpha);
    px[o] = (color[0] * alpha + px[o] * a * (1 - alpha)) / na;
    px[o + 1] = (color[1] * alpha + px[o + 1] * a * (1 - alpha)) / na;
    px[o + 2] = (color[2] * alpha + px[o + 2] * a * (1 - alpha)) / na;
    px[o + 3] = na * 255;
  };
  const edge = (M, i, x, y) => M[i] && (x === 0 || y === 0 || x === W - 1 || y === H - 1 || !M[i - 1] || !M[i + 1] || !M[i - W] || !M[i + W]);
  const heat = (v) => {
    // black -> blue -> yellow -> white, so 0.5 sits at a clear hue change.
    const t = v / 255;
    return t < 0.5 ? [0, 60 * t * 2, 255 * t * 2] : [255 * (t - 0.5) * 2, 60 + 195 * (t - 0.5) * 2, 255 - 255 * (t - 0.5) * 2];
  };
  for (let i = 0; i < N; i++) {
    const x = i % W, y = (i - x) / W;
    if (prob && P && P[i] > 8) blend(i, heat(P[i]), prob.alpha * Math.min(1, P[i] / 128));
    if (raw && MR && MR[i]) blend(i, raw.color, raw.alpha);
    if (adds && MF && MR && MF[i] && !MR[i]) blend(i, adds.color, adds.alpha);
    if (drops && ML && MF && MF[i] && !ML[i]) blend(i, drops.color, drops.alpha);
    if (residual && MX && T) {
      if (MX[i] && !T[i]) blend(i, [40, 80, 255], residual.alpha);
      else if (T[i] && !MX[i]) blend(i, [255, 50, 50], residual.alpha);
    }
    if (tubeFill && T && T[i]) blend(i, tubeFill.color, tubeFill.alpha);
    if (outline && MX && edge(MX, i, x, y)) blend(i, outline.color, outline.alpha);
    if (indep && TI && edge(TI, i, x, y)) blend(i, indep.color, indep.alpha * 0.8);
    if (tube && T && edge(T, i, x, y)) blend(i, tube.color, tube.alpha);
  }
  overlayCtx.putImageData(out, 0, 0);
  renderLegend();
}

function renderLegend() {
  const parts = [];
  for (const l of LAYERS) {
    if (!l.on || l.kind === "base") continue;
    if (l.id === "independent" && !(state.decoded && state.decoded.tube_independent) && !(state.frame && state.frame.pose && state.frame.pose.independent)) continue;
    if (l.id === "compare" && !state.comparePose) continue;
    if (l.id === "starts" && !state.starts) continue;
    if (l.id.startsWith("hyp_") && !(state.frame && state.frame.pose && state.frame.pose.hypotheses && state.frame.pose.hypotheses.some((h) => h.source === l.id.slice(4)))) continue;
    if (l.id === "prediction" && !(state.frame && state.frame.pose && state.frame.pose.prediction_xy)) continue;
    if (l.id === "cand_a" || l.id === "cand_b") {
      // Phase 3: the slot's legend entry names the shown set (regions.js).
      const text = typeof candidateSetLegendText === "function" ? candidateSetLegendText(l.id) : null;
      if (text) parts.push(`<span><span class="swatch" style="background:rgb(${l.color.join(",")})"></span>${escapeHtml(text)}</span>`);
      continue;
    }
    if (l.id === "residual") { parts.push(`<span><span class="swatch" style="background:rgb(40,80,255)"></span>mask missed</span><span><span class="swatch" style="background:rgb(255,50,50)"></span>tube extra</span>`); continue; }
    parts.push(`<span><span class="swatch" style="background:rgb(${l.color.join(",")})"></span>${l.name.replace(/ \(.*\)$/, "")}</span>`);
  }
  $("#legend").innerHTML = parts.join("");
  $("#legend").hidden = parts.length === 0;
}

// ---------------------------------------------------------------- stage drawing

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  if (!state.userView && state.decoded) fitView();
  else draw();
}

function fitView() {
  const d = state.decoded;
  if (!d || !d.width) return;
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / d.width, rect.height / d.height) * 0.98;
  state.view = { scale, tx: (rect.width - d.width * scale) / 2, ty: (rect.height - d.height * scale) / 2 };
  state.userView = false;
  draw();
}

function toImage(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const v = state.view;
  return { x: (clientX - rect.left - v.tx) / v.scale, y: (clientY - rect.top - v.ty) / v.scale };
}

function drawCurve(points, color, width, dash, alpha) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha === undefined ? 1 : alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width / state.view.scale;
  ctx.setLineDash(dash ? dash.map((v) => v / state.view.scale) : []);
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.stroke();
  ctx.restore();
}

function drawEnds(points, color) {
  const s = state.view.scale;
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 2 / s;
  const [hx, hy] = points[0];
  ctx.strokeStyle = "rgb(255,210,60)";
  ctx.strokeRect(hx - 5 / s, hy - 5 / s, 10 / s, 10 / s);
  const [tx, ty] = points[points.length - 1];
  ctx.strokeStyle = "rgb(80,200,255)";
  ctx.beginPath(); ctx.arc(tx, ty, 6 / s, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
}

function drawWidthTicks(points, profile, color, every = 4) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 1 / state.view.scale; ctx.globalAlpha = 0.9;
  ctx.beginPath();
  for (let i = 0; i < points.length; i += every) {
    const a = points[Math.max(0, i - 1)], b = points[Math.min(points.length - 1, i + 1)];
    let nx = -(b[1] - a[1]), ny = b[0] - a[0];
    const norm = Math.hypot(nx, ny) || 1;
    nx /= norm; ny /= norm;
    const half = profile[i] / 2;
    ctx.moveTo(points[i][0] - nx * half, points[i][1] - ny * half);
    ctx.lineTo(points[i][0] + nx * half, points[i][1] + ny * half);
  }
  ctx.stroke();
  ctx.restore();
}

function draw() {
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const d = state.decoded;
  if (!d || !d.width) { renderCaption(); return; }
  const v = state.view;
  ctx.setTransform(ratio * v.scale, 0, 0, ratio * v.scale, ratio * v.tx, ratio * v.ty);
  ctx.imageSmoothingEnabled = v.scale < 1;
  const base = layer("image");
  const image = state.showRaw && state.imageRaw ? state.imageRaw : state.image;
  if (base.on && image) { ctx.globalAlpha = base.alpha; ctx.drawImage(image, 0, 0); ctx.globalAlpha = 1; }
  else { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, d.width, d.height); }
  ctx.drawImage(overlayCanvas, 0, 0);
  const pose = state.frame && state.frame.pose;
  if (pose) {
    if (layer("crop").on && pose.crop) {
      const [x0, x1, y0, y1] = pose.crop;
      ctx.save(); ctx.strokeStyle = "rgba(160,160,160,0.8)"; ctx.setLineDash([6 / v.scale, 4 / v.scale]); ctx.lineWidth = 1 / v.scale;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.restore();
    }
    const indep = layer("independent");
    if (indep.on && pose.independent) {
      drawCurve(pose.independent.centerline_xy, `rgb(${indep.color.join(",")})`, 2, [8, 5], indep.alpha);
    }
    if (pose.hypotheses) {
      // Every candidate of the frame, by source; the path's choice wide underneath the centerline.
      for (const h of pose.hypotheses) {
        const l = layer(`hyp_${h.source}`);
        if (!l || !l.on) continue;
        const color = `rgb(${(HYP_COLORS[h.source] || [200, 200, 200]).join(",")})`;
        if (h.chosen) drawCurve(h.centerline_xy, color, 6, null, 0.35 * l.alpha);
        else drawCurve(h.centerline_xy, color, 1.2, h.source === "independent" ? [2, 3] : [6, 4], 0.8 * l.alpha);
      }
      const pred = layer("prediction");
      if (pred.on && pose.prediction_xy) {
        drawCurve(pose.prediction_xy, `rgb(${pred.color.join(",")})`, 1.5, [2, 4], pred.alpha);
        const [x, y] = pose.prediction_xy[0];
        ctx.save(); ctx.fillStyle = `rgba(255,255,255,${pred.alpha})`; ctx.beginPath(); ctx.arc(x, y, 3 / v.scale, 0, Math.PI * 2); ctx.fill(); ctx.restore();
      }
    }
    const cmp = layer("compare");
    if (cmp.on && state.comparePose && state.comparePose.pose) {
      drawCurve(state.comparePose.pose.centerline_xy, `rgb(${cmp.color.join(",")})`, 2, [3, 4], cmp.alpha);
      drawEnds(state.comparePose.pose.centerline_xy, `rgb(${cmp.color.join(",")})`);
    }
    // Phase 3: the chosen candidates of the shown candidate sets, under the centerline.
    if (typeof drawCandidateSetOverlays === "function") drawCandidateSetOverlays();
    if (layer("width_ticks").on && pose.width_profile) drawWidthTicks(pose.centerline_xy, pose.width_profile, "rgba(255,80,165,0.9)");
    const line = layer("centerline");
    if (line.on && pose.centerline_xy) {
      drawCurve(pose.centerline_xy, `rgb(${line.color.join(",")})`, 2, null, line.alpha);
      drawEnds(pose.centerline_xy, `rgb(${line.color.join(",")})`);
    }
  }
  const starts = layer("starts");
  if (starts.on && state.starts) {
    const palette = ["rgb(120,255,255)", "rgb(255,255,120)", "rgb(200,120,255)", "rgb(120,255,160)", "rgb(255,160,120)"];
    state.starts.forEach((s, i) => {
      drawCurve(s.centerline_xy, palette[i % palette.length], 1.5, [2, 3], starts.alpha);
      const [x, y] = s.centerline_xy[0];
      ctx.save(); ctx.fillStyle = palette[i % palette.length]; ctx.font = `${11 / v.scale}px system-ui`;
      ctx.fillText(s.name, x + 6 / v.scale, y - 6 / v.scale); ctx.restore();
    });
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  renderCaption();
}

function renderCaption() {
  const f = state.frame;
  if (!f || !state.run) { $("#caption").textContent = ""; return; }
  const s = f.stats || {};
  const parts = [`${state.run.entry.recording} · frame ${f.frame_index}`];
  if (s.fitted) {
    parts.push(`IoU ${fmt(s.iou)}`);
    if (s.iou_independent !== undefined && s.source) parts.push(`(indep ${fmt(s.iou_independent)})`);
    parts.push(`len ${fmt(s.body_length_px, 0)} px`, `width ${fmt(s.width_px, 1)} px`, `in view ${fmt(s.in_view_fraction, 2)}`);
    const prov = frameProvenance(f);
    if (prov && prov.algorithm) parts.push(shortAlgorithm(prov.algorithm) + (prov.edit ? " ✎" : ""));
    else if (s.source_name && s.source_name !== "independent") parts.push(s.source_name);
    if (s.ambiguity_score) parts.push(`score ${s.ambiguity_score}`);
  } else parts.push("no fit");
  if (f.threshold !== undefined && f.threshold !== state.run.threshold) parts.push(`threshold ${f.threshold} (override)`);
  $("#caption").textContent = parts.join("  ");
}

// ---------------------------------------------------------------- details panel

function row(label, value, cls) {
  return `<tr><td>${label}</td><td class="num ${cls || ""}">${value}</td></tr>`;
}
function section(label) { return `<tr class="section"><td colspan="2">${label}</td></tr>`; }

// Provenance of the frame's pose: from the frame payload when it carries it,
// else from the per-row arrays of the source payload.
function frameProvenance(f) {
  if (!f) return null;
  const direct = f.provenance || (f.stats && f.stats.provenance);
  if (direct && typeof direct === "object") return direct;
  const arrays = state.run && state.run.provenance;
  if (arrays && Array.isArray(arrays.algorithm) && f.stats && f.stats.row !== undefined) {
    const r = f.stats.row;
    return { algorithm: arrays.algorithm[r], job: arrays.job ? arrays.job[r] : undefined, time: arrays.time ? arrays.time[r] : undefined, edited: arrays.edited ? !!arrays.edited[r] : undefined };
  }
  const series = state.run && state.run.series;
  if (series && series.provenance_algorithm && f.stats && f.stats.row !== undefined) {
    const r = f.stats.row;
    return { algorithm: series.provenance_algorithm[r], job: series.provenance_job ? series.provenance_job[r] : undefined, time: series.provenance_time ? series.provenance_time[r] : undefined };
  }
  return null;
}

function renderDetails() {
  const f = state.frame;
  if (!f) return;
  const s = f.stats || {};
  const c = s.classification || { label: s.fitted ? "fitted" : "unfitted", kind: s.fitted ? "clean" : "unfitted", tags: [] };
  const badge = $("#classification");
  badge.textContent = c.label;
  badge.className = `badge ${c.kind}`;
  $("#tags").innerHTML = (c.tags || []).map((t) => `<span>${escapeHtml(t)}</span>`).join("");
  const cmp = state.comparePose && state.comparePose.stats;
  const both = (key, digits) => cmp ? `${fmt(s[key], digits)} <span style="color:#ffaa3c">/ ${fmt(cmp[key], digits)}</span>` : fmt(s[key], digits);
  const nPoints = state.run.n_points || 100;
  const rows = [];
  rows.push(section("fit" + (cmp ? " (this / compare)" : "")));
  rows.push(row("fitted", s.fitted ? "yes" : "no"));
  rows.push(row("IoU", both("iou")));
  if (s.tube_coverage !== undefined) rows.push(row("tube covered by mask", fmt(s.tube_coverage), s.iou !== null && s.iou < 0.9 && s.tube_coverage >= 0.9 ? "diff" : ""));
  if (s.iou_independent !== undefined) rows.push(row("IoU independent", fmt(s.iou_independent)));
  rows.push(row("source", s.source_name || "independent"), row("best start", s.best_start || "–", "wrap"), row("starts tried", fmt(s.n_starts)));
  rows.push(row("soft-Dice energy", fmt(s.energy, 4)), row("total energy", fmt(s.total_energy, 4)));
  if (s.stretch) rows.push(row("stretch", `#${s.stretch.index + 1} frames ${s.stretch.frames[0]}–${s.stretch.frames[1]} (${s.stretch.length})`));
  const prov = frameProvenance(f);
  if (prov) {
    rows.push(section("provenance"));
    rows.push(row("algorithm", escapeHtml(prov.algorithm || "unknown"), prov.algorithm ? "" : "dim"));
    rows.push(row("job / edit", escapeHtml(prov.job || "–"), "wrap"));
    rows.push(row("time", escapeHtml(fmtTime(prov.time))));
    if (prov.params) rows.push(row("params", escapeHtml(JSON.stringify(prov.params))));
    const edit = prov.edit;
    if (edit) rows.push(row("edit", `${escapeHtml(edit.id)} · ${escapeHtml(editLabel(edit))}${edit.note ? " — " + escapeHtml(edit.note) : ""} · ${escapeHtml(fmtTime(edit.time))}`, "wrap edited"));
    else if (prov.edited) rows.push(row("edit", "touched by a manual edit", "edited"));
    else if (edit === null && isWorkspace()) rows.push(row("edit", "none", "dim"));
  }
  if (f.pose && f.pose.hypotheses) {
    const path = f.pose.path || {};
    rows.push(section(`path (6c) · ${f.pose.hypotheses.length} candidates${path.mirrored ? " · chosen mirrored" : ""}`));
    rows.push(row("path", path.override ? `overrode lowest energy by ${fmt(path.energy_gap, 4)}` : "lowest energy", path.override ? "diff" : ""));
    rows.push(row("distance to prediction", f.pose.prediction_distance_px === null || f.pose.prediction_distance_px === undefined ? "–" : `${fmt(f.pose.prediction_distance_px, 1)} px`));
  }
  rows.push(section("body"));
  rows.push(row("length px", both("body_length_px", 1)));
  if (s.length_vs_prior_sigmas !== undefined) rows.push(row("length vs prior", `${fmt(s.length_vs_prior_sigmas, 2)} σ`, Math.abs(s.length_vs_prior_sigmas) > 2 ? "diff" : ""));
  rows.push(row("width px", both("width_px", 2)));
  if (s.width_vs_prior_sigmas !== undefined) rows.push(row("width vs prior", `${fmt(s.width_vs_prior_sigmas, 2)} σ`));
  rows.push(row("in view", `${fmt(s.points_in_fov)}/${nPoints} (${fmt(s.in_view_fraction, 2)})`));
  rows.push(row("taper asymmetry", fmt(s.taper_asymmetry, 3)), row("orientation gap", fmt(s.orientation_gap, 4)), row("reversed start won", s.reversed ? "yes" : "no"));
  if (s.max_bend_widths !== undefined) rows.push(row("tightest bend (width/radius)", fmt(s.max_bend_widths, 2), s.max_bend_widths > 2 ? "diff" : ""));
  if (s.track_length_px !== undefined && s.track_length_px !== null) rows.push(row("track length px", `${fmt(s.track_length_px, 1)}${s.length_refit ? " · refit" : ""}`));
  rows.push(row("tube area px", fmt(s.tube_area_px, 0)));
  if (s.crop) rows.push(row("crop x0 x1 y0 y1", s.crop.join(" ")));
  rows.push(section("mask (stored)"));
  rows.push(row("worm px", fmt(s.worm_pixels)), row("raw px", fmt(s.raw_worm_pixels)), row("hole fill px", fmt(s.pixels_filled)));
  rows.push(row("components", fmt(s.components)), row("outside largest px", fmt(s.pixels_outside_largest)), row("mask on border", s.mask_on_border ? "yes" : "no"));
  rows.push(section("ambiguity signals"));
  rows.push(row("area ratio mask/tube", fmt(s.area_ratio, 3)), row("self-contact px", fmt(s.self_contact_px, 1)), row("pose jump px", fmt(s.pose_jump_px, 1)));
  rows.push(row("length deviation (log)", fmt(s.length_deviation, 4)));
  rows.push(row("score", fmt(s.ambiguity_score)));
  if (s.score_independent !== undefined) rows.push(row("score independent", fmt(s.score_independent)));
  $("#stats").innerHTML = rows.join("");

  const flags = (s.flags || []).map((fl) => {
    const test = fl.threshold === null || fl.threshold === undefined ? fmt(fl.value) : `${fmt(fl.value)} ${fl.test} ${fmt(fl.threshold)}`;
    return `<tr class="${fl.fired ? "fired" : ""} ${fl.group}" title="${escapeHtml(fl.description)}"><td><span class="dot"></span>${fl.name}</td><td class="num">${test}</td><td>${fl.group}</td></tr>`;
  });
  $("#flags").innerHTML = flags.join("");

  const m = f.mask_stats, st = f.mask_stats_stored || {};
  const mrows = [];
  if (m) {
    const pair = (label, key) => {
      const differs = st[key] !== undefined && st[key] !== m[key];
      mrows.push(`<tr><td>${label}</td><td class="num">${fmt(m[key])}</td><td class="num ${differs ? "diff" : ""}">${st[key] === undefined ? "" : "stored " + fmt(st[key])}</td></tr>`);
    };
    mrows.push(`<tr><td>threshold</td><td class="num">${f.threshold}</td><td class="num">${f.threshold !== state.run.threshold ? "run " + state.run.threshold : ""}</td></tr>`);
    pair("raw mask px", "raw_worm_pixels"); pair("hole fill adds px", "pixels_filled"); pair("components", "components");
    pair("outside largest px", "pixels_outside_largest"); pair("final mask px", "worm_pixels");
    if (f.has_stored_mask !== undefined) mrows.push(`<tr><td>mask source</td><td colspan="2" class="num">${f.has_stored_mask ? "stored in the workspace" : "recomputed by the server"}</td></tr>`);
    else if (f.mask_source) mrows.push(`<tr><td>mask source</td><td colspan="2" class="num">${escapeHtml(f.mask_source)}</td></tr>`);
    const entry = state.run.entry;
    if (f.checkpoint && entry.checkpoint_path && !f.checkpoint.endsWith(entry.checkpoint_path.split("/").slice(-2).join("/")) && f.checkpoint !== entry.checkpoint_path) {
      mrows.push(`<tr><td colspan="3" class="diff">segmenter differs from the run's: ${escapeHtml(f.checkpoint)}</td></tr>`);
    }
  } else if (f.detail === "light") mrows.push('<tr><td colspan="3">mask layers loading…</td></tr>');
  else mrows.push(`<tr><td colspan="3">no mask layers (${escapeHtml((f.errors || []).join("; ") || "recording or checkpoint unavailable")})</td></tr>`);
  $("#mask-stats").innerHTML = mrows.join("");

  drawWidthChart();
  drawCurvatureChart();
  renderLayerAvailability();
  renderHypothesesTable();
  renderSegmentInfo();
  if (typeof renderFrameCandidateSets === "function") renderFrameCandidateSets();
}

// ---------------------------------------------------------------- small charts

// A chart canvas is laid out at its CSS size and backed at device pixels;
// drawing happens in CSS pixels through the transform.
function chartBox(canvasNode) {
  const ratio = window.devicePixelRatio || 1;
  // Setting canvas.height rewrites the height attribute, so the intended CSS
  // height lives in data-height and is never read back from the element.
  if (!canvasNode.dataset.height) canvasNode.dataset.height = canvasNode.getAttribute("height");
  const cssHeight = Number(canvasNode.dataset.height);
  if (canvasNode.style.height !== `${cssHeight}px`) canvasNode.style.height = `${cssHeight}px`;
  const rect = canvasNode.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(cssHeight * ratio));
  if (canvasNode.width !== width) canvasNode.width = width;
  if (canvasNode.height !== height) canvasNode.height = height;
  const c = canvasNode.getContext("2d");
  c.setTransform(ratio, 0, 0, ratio, 0, 0);
  c.clearRect(0, 0, rect.width, cssHeight);
  return { c, w: rect.width, h: cssHeight };
}

function polyline(c, xs, ys, color, dash, width = 1.5) {
  c.save(); c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash || []);
  c.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] === null || ys[i] === undefined || Number.isNaN(ys[i])) { started = false; continue; }
    if (!started) { c.moveTo(xs[i], ys[i]); started = true; } else c.lineTo(xs[i], ys[i]);
  }
  c.stroke(); c.restore();
}

function drawWidthChart() {
  const node = $("#width-chart");
  const { c, w, h } = chartBox(node);
  const pose = state.frame && state.frame.pose;
  if (!pose || !pose.width_profile) return;
  const series = [
    { y: pose.width_profile, color: "#57d68d", dash: null, width: 2 },
    { y: pose.width_template_profile, color: "#9aa6b0", dash: [2, 3] },
    { y: pose.width_prior_profile, color: "#9aa6b0", dash: [6, 4] },
    { y: pose.independent && pose.independent.width_profile, color: "#cccccc", dash: [8, 5] },
    { y: state.comparePose && state.comparePose.pose && state.comparePose.pose.width_profile, color: "#ffaa3c", dash: [3, 4] },
  ].filter((s) => s.y);
  const max = Math.max(...series.flatMap((s) => s.y)) * 1.1 || 1;
  const pad = { l: 30, r: 6, t: 6, b: 16 };
  const n = pose.width_profile.length;
  const X = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - v / max) * (h - pad.t - pad.b);
  c.fillStyle = "#9aa6b0"; c.font = "10px system-ui";
  for (const tick of [0, max / 2, max]) { c.fillText(tick.toFixed(0), 2, Y(tick) + 3); c.strokeStyle = "#1f2a33"; c.beginPath(); c.moveTo(pad.l, Y(tick)); c.lineTo(w - pad.r, Y(tick)); c.stroke(); }
  c.fillText("head", pad.l, h - 4); c.fillText("tail", w - pad.r - 18, h - 4);
  const xs = Array.from({ length: n }, (_, i) => X(i));
  for (const s of series) polyline(c, xs, s.y.map(Y), s.color, s.dash, s.width || 1.2);
  // Shade the part of the body outside the camera, when known.
  const shape = state.run.image_shape;
  if (shape && pose.centerline_xy) {
    c.fillStyle = "rgba(128,200,255,0.15)";
    pose.centerline_xy.forEach(([x, y], i) => {
      if (x < 0 || y < 0 || x >= shape[1] || y >= shape[0]) c.fillRect(X(i) - 0.5, pad.t, Math.max(1, (w - pad.l - pad.r) / n), h - pad.t - pad.b);
    });
  }
}

function drawCurvatureChart() {
  const node = $("#curvature-chart");
  const { c, w, h } = chartBox(node);
  const pose = state.frame && state.frame.pose;
  if (!pose || !pose.curvature) return;
  const k = pose.curvature;
  const limit = 2 / Math.max(pose.width_px, 1);  // the fitter's bend limit: radius of half a width
  const max = Math.max(limit * 1.5, ...k.map(Math.abs)) * 1.05;
  const pad = { l: 40, r: 6, t: 4, b: 4 };
  const n = k.length;
  const X = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (0.5 - v / (2 * max)) * (h - pad.t - pad.b);
  c.fillStyle = "rgba(255,93,93,0.12)";
  c.fillRect(pad.l, pad.t, w - pad.l - pad.r, Y(limit) - pad.t);
  c.fillRect(pad.l, Y(-limit), w - pad.l - pad.r, h - pad.b - Y(-limit));
  c.strokeStyle = "#2a343c"; c.beginPath(); c.moveTo(pad.l, Y(0)); c.lineTo(w - pad.r, Y(0)); c.stroke();
  c.fillStyle = "#9aa6b0"; c.font = "10px system-ui";
  c.fillText(`+${max.toFixed(3)}`, 2, pad.t + 9); c.fillText(`−${max.toFixed(3)}`, 2, h - pad.b - 2); c.fillText("limit", 2, Y(limit) + 3);
  const xs = Array.from({ length: n }, (_, i) => X(i));
  polyline(c, xs, k.map(Y), "#57d68d", null, 1.5);
  if (state.comparePose && state.comparePose.pose && state.comparePose.pose.curvature) polyline(c, xs, state.comparePose.pose.curvature.map(Y), "#ffaa3c", [3, 4], 1);
}

// ---------------------------------------------------------------- layers UI

function renderLayers() {
  const box = $("#layers");
  box.innerHTML = "";
  LAYERS.forEach((l, index) => {
    const node = document.createElement("div");
    node.className = "layer";
    node.dataset.layer = l.id;
    const key = index < 9 ? `${index + 1}` : "";
    node.innerHTML = `<input type="checkbox" ${l.on ? "checked" : ""} title="${key}"><span><span class="swatch" style="background:${l.color ? `rgb(${l.color.join(",")})` : "#888"}"></span>${l.name} <span class="key">${key}</span></span><input type="range" min="0" max="1" step="0.05" value="${l.alpha}" title="opacity">`;
    node.querySelector("input[type=checkbox]").addEventListener("change", (e) => { l.on = e.target.checked; relayer(l); });
    node.querySelector("input[type=range]").addEventListener("input", (e) => { l.alpha = parseFloat(e.target.value); relayer(l); });
    box.appendChild(node);
  });
}

function relayer(l) {
  if (l.kind === "pixel" || l.kind === "both") buildOverlay();
  renderLegend();
  draw();
}

function toggleLayer(index) {
  const l = LAYERS[index];
  if (!l) return;
  l.on = !l.on;
  const node = $(`.layer[data-layer="${l.id}"] input[type=checkbox]`);
  if (node) node.checked = l.on;
  relayer(l);
}

function renderLayerAvailability() {
  const d = state.decoded || {};
  const pose = state.frame && state.frame.pose;
  if (state.frame && state.frame.detail === "light") {
    const hyp = (src) => pose && pose.hypotheses && pose.hypotheses.some((h) => h.source === src);
    const light = { independent: pose && pose.independent, compare: state.comparePose, starts: state.starts, hyp_forward: hyp("forward"), hyp_backward: hyp("backward"), hyp_independent: hyp("independent"), prediction: pose && pose.prediction_xy, cand_a: !!state.shownSets[0], cand_b: !!state.shownSets[1] };
    for (const node of document.querySelectorAll(".layer")) node.classList.toggle("unavailable", node.dataset.layer in light && !light[node.dataset.layer]);
    return;
  }
  const available = {
    probability: !!d.probability, mask_raw: !!d.mask_raw, fill_adds: !!d.mask_filled, largest_drops: !!d.mask_largest,
    residual: !!(d.mask_final && d.tube), tube: !!d.tube, tube_fill: !!d.tube, final_outline: !!d.mask_final,
    centerline: !!pose, width_ticks: !!pose, crop: !!pose,
    independent: !!(pose && pose.independent), compare: !!state.comparePose, starts: !!state.starts, image: true,
    hyp_forward: !!(pose && pose.hypotheses && pose.hypotheses.some((h) => h.source === "forward")),
    hyp_backward: !!(pose && pose.hypotheses && pose.hypotheses.some((h) => h.source === "backward")),
    hyp_independent: !!(pose && pose.hypotheses && pose.hypotheses.some((h) => h.source === "independent")),
    prediction: !!(pose && pose.prediction_xy),
    cand_a: !!state.shownSets[0], cand_b: !!state.shownSets[1],
  };
  for (const node of document.querySelectorAll(".layer")) node.classList.toggle("unavailable", available[node.dataset.layer] === false);
}
