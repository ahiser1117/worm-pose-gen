"use strict";

// Body fields screen: browse the corpus store's body-field targets
// (/api/body-fields, app/routers/body_fields.py), show them over the labeled
// frame or its temporal context (the frames t-16..t+16 and the symmetric
// differences the network sees), and correct them: flip head/tail, review,
// trace the midline (clicked points, head first) to refit a wrong route, and
// rebuild after a mask edit.  The mask itself is edited in Paint (paintScreen).
const bodyFields = (() => {
  const LAYER_DEFS = [
    {id: "mask", name: "Hand mask", on: false, alpha: 0.35, swatch: "rgb(255,65,155)"},
    {id: "ap", name: "A-P field (head 0 → tail 1)", on: true, alpha: 0.65, swatch: "linear-gradient(90deg,#440154,#21918c,#fde725)"},
    {id: "overlap", name: "Overlap (crossing)", on: true, alpha: 1.0, swatch: "#fff"},
    {id: "tube", name: "Fitted tube outline", on: true, alpha: 0.9, swatch: "rgb(90,220,140)"},
    {id: "centerline", name: "Fitted centerline", on: false, alpha: 0.9, swatch: "rgb(255,80,165)"},
    {id: "ends", name: "Head (green) / tail (red)", on: true, alpha: 1.0, swatch: "linear-gradient(90deg,#3f3 50%,#f44 50%)"},
    {id: "nose", name: "Acquisition nose ✕", on: true, alpha: 1.0, swatch: "rgb(255,210,60)"},
    {id: "trace", name: "Stored trace (numbered clicks)", on: true, alpha: 0.9, swatch: "rgb(255,150,40)"},
  ];
  const TRACE_METHODS = {traced: "traced", trace_as_drawn: "trace as drawn"};
  const VIRIDIS = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
  const layers = LAYER_DEFS.map(l => ({...l}));
  let rows = [], detail = null, context = null, overlays = {}, request = 0, contextRequest = 0, busy = false;
  // Trace mode: the clicked points (head first) and the preview fit along them, if any.
  let trace = null;
  let contextNote = "", view = {scale: 1, tx: 0, ty: 0}, drag = null, mode = "frame", offset = 0, lag = 4, gain = 2, playTimer = null;
  const jobStates = new Map();
  const node = id => document.getElementById(id);
  const canvas = () => node("bf-canvas");
  const baseCanvas = document.createElement("canvas");
  const maxLag = () => detail?.max_lag ?? 16;
  const id = () => detail?.sample.sample_id;

  function viridis(t) {
    const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1), i = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - i;
    return VIRIDIS[i].map((v, k) => Math.round(v + (VIRIDIS[i + 1][k] - v) * f));
  }
  function paintCanvas(width, height, fill) {
    const out = document.createElement("canvas"); out.width = width; out.height = height;
    const c = out.getContext("2d"), image = c.createImageData(width, height);
    for (let i = 0; i < width * height; i++) { const rgba = fill(i); if (rgba) image.data.set(rgba, i * 4); }
    c.putImageData(image, 0, 0); return out;
  }
  const status = (message, kind) => { const n = node("bf-edit-status"); if (n) { n.textContent = message; n.className = "note" + (kind ? " " + kind : ""); } };

  // ------------------------------------------------------------ list

  function filters() {
    const ids = {split: "bf-split", orientation: "bf-orientation", method: "bf-method", review: "bf-review", status: "bf-status", contact: "bf-contact", min_iou: "bf-min-iou", max_iou: "bf-max-iou"};
    return Object.fromEntries(Object.entries(ids).map(([key, n]) => [key, node(n)?.value.trim() || ""]).filter(([, v]) => v));
  }
  function fill(idName, values, label) {
    const select = node(idName), previous = select.value;
    select.replaceChildren(new Option(label, ""), ...values.map(v => new Option(v, v)));
    if (previous && !values.includes(previous)) select.add(new Option(previous, previous));
    select.value = previous;
  }
  function rowText(row) {
    const bits = [`frame ${row.frame_index}`, row.split];
    if (row.status === "missing") bits.push("no body fields");
    else if (!row.has_body) bits.push("no body");
    else bits.push(row.orientation, `IoU ${row.fit_iou.toFixed(3)}${row.auto_fit_iou != null && TRACE_METHODS[row.fit_method] ? ` (auto ${row.auto_fit_iou.toFixed(3)})` : ""}`);
    if (row.self_contact) bits.push("contact");
    if (row.overlap_px) bits.push(`overlap ${row.overlap_px} px`);
    return bits.join(" · ");
  }
  function renderList() {
    const box = node("bf-list"); box.replaceChildren();
    for (const row of rows) {
      const item = document.createElement("div"); item.className = "item bf-row"; item.dataset.sample = row.sample_id;
      item.classList.toggle("current", row.sample_id === id());
      const text = document.createElement("span"); text.textContent = row.recording;
      const meta = document.createElement("div"); meta.className = "meta"; meta.textContent = rowText(row); text.append(meta);
      const badges = document.createElement("span"); badges.className = "bf-badges";
      for (const [label, kind] of [[row.status !== "current" ? row.status : "", row.status], [row.review && row.review !== "unreviewed" ? row.review : "", row.review], [TRACE_METHODS[row.fit_method] || "", "traced"], [row.job ? "rebuilding" : "", "job"]]) {
        if (!label) continue;
        const badge = document.createElement("b"); badge.className = `bf-badge ${kind}`; badge.textContent = label; badges.append(badge);
      }
      item.append(text, badges); item.onclick = () => open(row.sample_id);
      box.append(item);
    }
    if (!rows.length) { const empty = document.createElement("div"); empty.className = "empty"; empty.textContent = "No samples match these filters."; box.append(empty); }
    box.querySelector(".current")?.scrollIntoView({block: "nearest"});
  }
  async function refresh() {
    const query = new URLSearchParams(filters()).toString();
    try {
      const payload = await api(`/api/body-fields${query ? "?" + query : ""}`);
      rows = payload.samples;
      fill("bf-orientation", payload.facets.orientations, "All orientations");
      fill("bf-method", payload.facets.methods, "All fit methods");
      const c = payload.counts;
      node("bf-counts").textContent = `${rows.length} shown / ${payload.total} samples · ${c.status.current} current · ${c.status.stale} stale · ${c.status.missing} missing · ${c.review.accepted} accepted · ${c.review.rejected} rejected\n${payload.root}`;
      renderList();
    } catch (error) { node("bf-counts").textContent = error.message; }
  }

  // ------------------------------------------------------------ sample

  async function open(sampleId, {keepView = false} = {}) {
    const token = ++request;
    try {
      const payload = await api(`/api/body-fields/${encodeURIComponent(sampleId)}`);
      const [mask, ap, overlap, image] = await Promise.all([decodeGray(payload.mask), decodeGray(payload.ap), decodeGray(payload.overlap), loadImage(payload.image)]);
      if (token !== request) return false;
      const sameSample = detail && detail.sample.sample_id === sampleId;
      const {width, height} = payload;
      detail = {...payload, _image: image, _mask: mask, _ap: ap, _overlap: overlap};
      overlays = {
        mask: paintCanvas(width, height, i => mask[i] === 255 ? [255, 65, 155, 255] : mask[i] === 128 ? [255, 205, 40, 255] : null),
        ap: ap && paintCanvas(width, height, i => ap[i] ? [...viridis((ap[i] - 1) / 254), 255] : null),
        overlap: overlap && paintCanvas(width, height, i => overlap[i] ? [255, 255, 255, 255] : null),
      };
      if (!sameSample) { context = null; offset = 0; stopPlay(); status(""); trace = null; renderTrace(); }
      if (!keepView && !sameSample) fitView();
      renderBase(); renderSummary(); renderList(); syncContextControls(); draw();
      if (!sameSample && payload.meta) loadContext();
      return true;
    } catch (error) { if (token === request) status(error.message, "error"); return false; }
  }
  async function loadContext() {
    if (!detail?.meta) return;
    const token = ++contextRequest, sampleId = id();
    contextNote = "Loading context frames…"; syncContextControls();
    try {
      const payload = await api(`/api/body-fields/${encodeURIComponent(sampleId)}/context`);
      const gray = await Promise.all(payload.frames.map(decodeGray));
      if (token !== contextRequest || sampleId !== id()) return;
      context = {valid: payload.valid, gray};
      lag = Math.min(lag, payload.max_lag); offset = Math.max(-payload.max_lag, Math.min(payload.max_lag, offset));
      renderBase(); syncContextControls(); draw();
    } catch (error) { if (token === contextRequest) { contextNote = error.message; syncContextControls(); } }
  }
  function summaryLines() {
    const s = detail.sample, m = detail.meta, lines = [`${s.recording} · frame ${s.frame_index} · ${s.split} · mask revision ${s.mask_revision}`];
    if (!m) lines.push("No body fields yet: rebuild to fit the tube and render the targets.");
    else {
      if (s.status === "stale") lines.push(`STALE: built from mask revision ${m.mask_revision}; the label has changed since. Rebuild.`);
      if (!m.has_body) lines.push("The label holds no worm: context only, trains the mask.");
      else {
        const orientation = m.orientation === "nose_nearby" ? `nose landmark ${m.nose_offset > 0 ? "+" : ""}${m.nose_offset} frames away` : m.orientation === "nose" ? "nose landmark on this frame" : m.orientation === "taper" ? "taper (no nose landmark)" : "set by hand";
        lines.push(`Head from ${orientation}${m.orientation_margin !== undefined ? ` · margin ${m.orientation_margin.toFixed(2)} diameters` : ""}`);
        lines.push(`Tube fit IoU ${m.fit_iou.toFixed(3)} · overlap ${m.overlap_px} px · diameter ${detail.diameter_px.toFixed(1)} px${s.self_contact ? " · self-contact" : ""}`);
        if (TRACE_METHODS[m.fit_method]) lines.push(`Fit ${TRACE_METHODS[m.fit_method]} along ${detail.trace_xy?.length ?? "?"} clicked points${m.auto_fit_iou != null ? ` · automatic fit IoU was ${m.auto_fit_iou.toFixed(3)}` : ""}`);
        else if (m.fit_method) lines.push(`Fit: ${m.fit_method}`);
      }
      lines.push(`Review: ${s.review}${m.reviewed_at ? " at " + m.reviewed_at : ""}${s.review === "rejected" ? " (trains the mask only)" : ""}`);
    }
    const job = activeJob();
    if (job) lines.push(`Rebuild job ${job.id} ${job.state}…`);
    return lines;
  }
  function activeJob() {
    return (state.jobs || []).find(j => (j.spec || {}).kind === "body_fields" && j.spec.params?.sample_id === id() && jobActive(j)) || (detail?.sample.job ? {id: detail.sample.job, state: "queued"} : null);
  }
  function renderSummary() {
    const box = node("bf-summary");
    box.textContent = detail ? summaryLines().join("\n") : "Choose a sample from the list.";
    box.classList.toggle("stale", detail?.sample.status !== "current");
    const has = !!detail?.meta, body = has && detail.meta.has_body, pending = !!activeJob();
    node("bf-flip").disabled = busy || pending || !body || !!trace;
    node("bf-trace").disabled = busy || pending || !body;
    node("bf-trace").classList.toggle("active", !!trace);
    for (const button of document.querySelectorAll("[data-bf-review]")) {
      button.disabled = busy || pending || !has;
      button.classList.toggle("active", has && detail.sample.review === button.dataset.bfReview);
    }
    node("bf-rebuild").disabled = busy || pending || !detail;
    node("bf-edit-mask").disabled = busy || !detail;
    node("bf-prev").disabled = node("bf-next").disabled = !rows.length;
    node("bf-caption").textContent = detail ? `${detail.sample.sample_id}${rows.length ? ` · ${rows.findIndex(r => r.sample_id === id()) + 1} of ${rows.length}` : ""}` : "";
  }
  function step(delta) {
    if (!rows.length) return;
    const index = rows.findIndex(r => r.sample_id === id());
    const next = index < 0 ? 0 : Math.max(0, Math.min(rows.length - 1, index + delta));
    if (rows[next].sample_id !== id()) open(rows[next].sample_id, {keepView: true});
  }

  // ------------------------------------------------------------ edits

  async function edit(label, run) {
    if (!detail || busy) return;
    busy = true; renderSummary(); status(`${label}…`);
    try { await run(); status(`${label}: done.`, "ok"); }
    catch (error) { status(`${label}: ${error.message}`, "error"); }
    finally { busy = false; renderSummary(); renderTrace(); }
  }
  const flip = () => edit("Flip head/tail", async () => {
    const payload = await post(`/api/body-fields/${encodeURIComponent(id())}/flip`, {});
    await open(payload.sample.sample_id, {keepView: true});
    await refresh();
  });
  const review = value => edit(`Review ${value}`, async () => {
    const sampleId = id();
    const payload = await post(`/api/body-fields/${encodeURIComponent(sampleId)}/review`, {status: value});
    detail.sample = payload.sample; detail.meta = payload.meta;
    await refresh();
    if (node("bf-advance").checked && value !== "unreviewed") step(1);
  });
  const rebuild = () => edit("Rebuild", async () => {
    const job = await post(`/api/body-fields/${encodeURIComponent(id())}/rebuild`, {});
    detail.sample.job = job.id;
    status(`Rebuild job ${job.id} queued: the tube is refit to the current mask. Progress and log are in Jobs.`);
    if (typeof loadJobs === "function") await loadJobs();
    await refresh();
  });
  async function editMask() {
    if (!detail) return;
    const opened = await paintScreen.openSamples([id()], {back: "bodyfields"});
    if (opened) setStatus("Editing the corpus label in Paint. Save it, return to Body fields and rebuild the stale targets.", "ok");
  }
  function onJobs() {
    let changed = false;
    for (const job of state.jobs || []) {
      if ((job.spec || {}).kind !== "body_fields") continue;
      const before = jobStates.get(job.id);
      jobStates.set(job.id, job.state);
      if (before === undefined || before === job.state || jobActive(job)) continue;
      changed = true;
      if (job.spec.params?.sample_id === id()) {
        status(job.state === "done" ? "Rebuild finished: targets refit to the current mask." : `Rebuild ${job.state}${job.error ? ": " + job.error : ""}`, job.state === "done" ? "ok" : "error");
        if (detail) detail.sample.job = null;
        context = null; open(id(), {keepView: true}).then(loadContext);
      }
    }
    if (changed && state.screen === "bodyfields") refresh();
    else if (detail) renderSummary();
  }

  // ------------------------------------------------------------ trace

  function setTrace(value) { trace = value; renderTrace(); renderSummary(); draw(); }
  function toggleTrace() {
    if (trace) { setTrace(null); status(""); return; }
    if (node("bf-trace").disabled) return;
    if (mode !== "frame" || offset) { mode = "frame"; offset = 0; stopPlay(); renderBase(); syncContextControls(); }
    setTrace({points: [], preview: null});
  }
  function addPoint(p) {
    if (p.x < 0 || p.y < 0 || p.x >= detail.width || p.y >= detail.height) return;
    trace.points.push([Math.round(p.x * 10) / 10, Math.round(p.y * 10) / 10]); trace.preview = null; renderTrace(); draw();
  }
  function removePoint() { if (trace?.points.length) { trace.points.pop(); trace.preview = null; renderTrace(); draw(); } }
  function discardPreview() { if (trace) { trace.preview = null; renderTrace(); draw(); } }
  const fitTrace = (asDrawn = false) => {
    if (!trace || trace.points.length < 2) { status("Click at least two points, head first.", "error"); return Promise.resolve(); }
    const points = trace.points.map(p => [...p]);
    return edit(asDrawn ? "Trace as drawn" : "Fit along the trace", async () => {
      const payload = await post(`/api/body-fields/${encodeURIComponent(id())}/trace`, {points, as_drawn: asDrawn, commit: false});
      const [ap, overlap] = await Promise.all([decodeGray(payload.ap), decodeGray(payload.overlap)]);
      if (!trace || JSON.stringify(trace.points) !== JSON.stringify(points)) return;
      trace.preview = {...payload, asDrawn, points, _ap: ap,
        _overlays: {ap: ap && paintCanvas(detail.width, detail.height, i => ap[i] ? [...viridis((ap[i] - 1) / 254), 255] : null),
                    overlap: overlap && paintCanvas(detail.width, detail.height, i => overlap[i] ? [255, 255, 255, 255] : null)}};
      renderTrace(); draw();
    });
  };
  const acceptTrace = () => {
    const preview = trace?.preview; if (!preview) return Promise.resolve();
    return edit("Accept the traced fit", async () => {
      await post(`/api/body-fields/${encodeURIComponent(id())}/trace`, {points: preview.points, as_drawn: preview.asDrawn, commit: true});
      trace = null; renderTrace();
      await open(id(), {keepView: true}); await refresh();
    });
  };
  function renderTrace() {
    const panel = node("bf-trace-panel"); panel.hidden = !trace;
    canvas()?.classList.toggle("tracing", !!trace);
    if (!trace) return;
    const preview = trace.preview, n = trace.points.length;
    node("bf-trace-count").textContent = n ? `${n} point${n === 1 ? "" : "s"}; point 1 is the head.` : "No points yet.";
    node("bf-trace-fit").disabled = busy || n < 2;
    node("bf-trace-undo").disabled = busy || !n;
    const result = node("bf-trace-result"); result.hidden = !preview;
    if (preview) {
      const old = detail.meta.auto_fit_iou ?? detail.meta.fit_iou;
      result.textContent = `${preview.asDrawn ? "Trace as drawn" : "Fit along the trace"}: IoU new ${preview.meta.fit_iou.toFixed(3)} vs old ${old.toFixed(3)}\nThe preview's A-P field and cyan tube are shown; the record's tube stays green.`;
    }
    for (const id of ["bf-trace-accept", "bf-trace-discard"]) node(id).disabled = busy || !preview;
    node("bf-trace-drawn").disabled = busy || n < 2;
  }

  // ------------------------------------------------------------ context

  function syncContextControls() {
    const L = maxLag(), has = !!context;
    const range = node("bf-offset"); range.min = -L; range.max = L; range.value = offset; range.disabled = !has || mode !== "frame";
    const lags = node("bf-lag"); lags.max = L; lags.value = lag; lags.disabled = !has || mode !== "difference";
    node("bf-offset-value").textContent = `${offset > 0 ? "+" : ""}${offset}`;
    node("bf-lag-value").textContent = `±${lag}`;
    node("bf-play").disabled = !has || mode !== "frame";
    node("bf-play").textContent = playTimer ? "Pause" : "Play";
    for (const input of document.querySelectorAll("input[name=bf-mode]")) { input.checked = input.value === mode; input.disabled = !has && input.value === "difference"; }
    let note = "";
    if (!detail?.meta) note = detail ? "No context stored for this sample." : "";
    else if (!has) note = contextNote;
    else if (mode === "frame") note = context.valid[L + offset] ? `Frame t${offset ? (offset > 0 ? "+" : "") + offset : ""} (${detail.sample.frame_index + offset})` : `t${offset > 0 ? "+" : ""}${offset} is outside the recording or unreadable: shows the nearest readable frame.`;
    else note = context.valid[L + lag] && context.valid[L - lag] ? `frame[t+${lag}] − frame[t−${lag}] · grey 0, light = brighter later` : `Lag ${lag} has an invalid end: the network gets a zero channel.`;
    node("bf-context-status").textContent = note;
  }
  function renderBase() {
    if (!detail) return;
    const {width, height} = detail, L = maxLag();
    baseCanvas.width = width; baseCanvas.height = height;
    const c = baseCanvas.getContext("2d");
    if (!context || (mode === "frame" && offset === 0)) { c.drawImage(detail._image, 0, 0); return; }
    const image = c.createImageData(width, height), out = image.data;
    if (mode === "frame") {
      const gray = context.gray[L + offset];
      for (let i = 0; i < gray.length; i++) { const j = i * 4; out[j] = out[j + 1] = out[j + 2] = gray[i]; out[j + 3] = 255; }
    } else {
      const ok = context.valid[L + lag] && context.valid[L - lag], later = context.gray[L + lag], earlier = context.gray[L - lag];
      for (let i = 0; i < later.length; i++) {
        const j = i * 4, v = ok ? Math.max(0, Math.min(255, 128 + gain * (later[i] - earlier[i]))) : 128;
        out[j] = out[j + 1] = out[j + 2] = v; out[j + 3] = 255;
      }
    }
    c.putImageData(image, 0, 0);
  }
  function setMode(value) { mode = value; if (mode !== "frame") stopPlay(); renderBase(); syncContextControls(); draw(); }
  function setOffset(value) { offset = Math.max(-maxLag(), Math.min(maxLag(), value)); renderBase(); syncContextControls(); draw(); }
  function setLag(value) { lag = Math.max(1, Math.min(maxLag(), value)); renderBase(); syncContextControls(); draw(); }
  function stopPlay() { clearInterval(playTimer); playTimer = null; if (node("bf-play")) node("bf-play").textContent = "Play"; }
  function togglePlay() {
    if (playTimer) { stopPlay(); syncContextControls(); return; }
    if (!context || mode !== "frame") return;
    playTimer = setInterval(() => setOffset(offset >= maxLag() ? -maxLag() : offset + 1), 1000 / Number(node("bf-fps").value || 10));
    syncContextControls();
  }

  // ------------------------------------------------------------ drawing

  function fitView() {
    const c = canvas(); if (!c || !detail) return;
    const rect = c.getBoundingClientRect(), scale = Math.min(rect.width / detail.width, rect.height / detail.height) || 1;
    view = {scale, tx: (rect.width - detail.width * scale) / 2, ty: (rect.height - detail.height * scale) / 2};
  }
  function toImage(clientX, clientY) {
    const rect = canvas().getBoundingClientRect();
    return {x: (clientX - rect.left - view.tx) / view.scale, y: (clientY - rect.top - view.ty) / view.scale};
  }
  function resize() {
    const c = canvas(), ratio = window.devicePixelRatio || 1, rect = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(rect.width * ratio)); c.height = Math.max(1, Math.round(rect.height * ratio));
    draw();
  }
  function tubePolygon(points, widths) {
    const left = [], right = [];
    for (let i = 0; i < points.length; i++) {
      const a = points[Math.max(0, i - 1)], b = points[Math.min(points.length - 1, i + 1)];
      const dx = b[0] - a[0], dy = b[1] - a[1], norm = Math.hypot(dx, dy) || 1, r = widths[i] / 2;
      left.push([points[i][0] - dy / norm * r, points[i][1] + dx / norm * r]);
      right.push([points[i][0] + dy / norm * r, points[i][1] - dx / norm * r]);
    }
    return [...left, ...right.reverse()];
  }
  function path(c, points, close) {
    c.beginPath(); points.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); if (close) c.closePath();
  }
  function draw() {
    const c = canvas(); if (!c) return;
    const g = c.getContext("2d"), ratio = window.devicePixelRatio || 1;
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height);
    if (!detail) return;
    g.setTransform(ratio * view.scale, 0, 0, ratio * view.scale, ratio * view.tx, ratio * view.ty);
    g.imageSmoothingEnabled = false;
    g.drawImage(baseCanvas, 0, 0);
    const layer = name => layers.find(l => l.id === name);
    const shown = trace?.preview ? {...overlays, ...trace.preview._overlays} : overlays;
    for (const name of ["mask", "ap", "overlap"]) {
      const l = layer(name);
      if (l.on && shown[name]) { g.globalAlpha = l.alpha; g.drawImage(shown[name], 0, 0); }
    }
    g.globalAlpha = 1;
    const line = 1.5 / view.scale, center = detail.centerline_xy;
    if (center) {
      const tube = layer("tube"), spine = layer("centerline"), ends = layer("ends");
      if (tube.on) { g.globalAlpha = tube.alpha; g.strokeStyle = "rgb(90,220,140)"; g.lineWidth = line; path(g, tubePolygon(center, detail.width_profile), true); g.stroke(); }
      if (spine.on) { g.globalAlpha = spine.alpha; g.strokeStyle = "rgb(255,80,165)"; g.lineWidth = line; path(g, center, false); g.stroke(); }
      if (ends.on) {
        g.globalAlpha = ends.alpha;
        const radius = Math.max(3, detail.diameter_px * 0.3);
        for (const [point, colour] of [[detail.head_xy, "#33ff33"], [detail.tail_xy, "#ff4444"]]) {
          g.beginPath(); g.arc(point[0], point[1], radius, 0, Math.PI * 2); g.fillStyle = colour; g.fill();
          g.lineWidth = line; g.strokeStyle = "#000"; g.stroke();
        }
      }
    }
    const preview = trace?.preview;
    if (preview?.centerline_xy) {
      g.globalAlpha = 1; g.strokeStyle = "rgb(60,220,255)"; g.lineWidth = 2 * line; g.setLineDash([6 * line, 4 * line]);
      path(g, tubePolygon(preview.centerline_xy, preview.width_profile), true); g.stroke(); g.setLineDash([]);
      const radius = Math.max(3, preview.diameter_px * 0.3);
      for (const [point, colour] of [[preview.head_xy, "#33ff33"], [preview.tail_xy, "#ff4444"]]) {
        g.beginPath(); g.arc(point[0], point[1], radius, 0, Math.PI * 2); g.fillStyle = colour; g.fill();
        g.lineWidth = 2 * line; g.strokeStyle = "rgb(60,220,255)"; g.stroke();
      }
    }
    const stored = layer("trace");
    if (trace) drawPoints(g, trace.points, 1, line, true);
    else if (stored.on && detail.trace_xy) drawPoints(g, detail.trace_xy, stored.alpha, line, false);
    const nose = layer("nose");
    if (nose.on && detail.nose_xy) {
      const [x, y] = detail.nose_xy, size = Math.max(4, (detail.diameter_px || 10) * 0.5);
      g.globalAlpha = nose.alpha; g.strokeStyle = "rgb(255,210,60)"; g.lineWidth = 2 * line;
      g.beginPath(); g.moveTo(x - size, y - size); g.lineTo(x + size, y + size); g.moveTo(x + size, y - size); g.lineTo(x - size, y + size); g.stroke();
    }
    g.globalAlpha = 1;
  }
  // A numbered polyline: point 1 is the head.
  function drawPoints(g, points, alpha, line, active) {
    if (!points.length) return;
    g.globalAlpha = alpha; g.strokeStyle = "rgb(255,150,40)"; g.lineWidth = 2 * line;
    if (!active) g.setLineDash([5 * line, 4 * line]);
    path(g, points, false); g.stroke(); g.setLineDash([]);
    const radius = 7 * line;
    g.font = `${10 * line}px system-ui, sans-serif`; g.textAlign = "center"; g.textBaseline = "middle";
    points.forEach(([x, y], i) => {
      g.beginPath(); g.arc(x, y, radius, 0, Math.PI * 2); g.fillStyle = i === 0 ? "rgb(51,255,51)" : "rgb(255,150,40)"; g.fill();
      g.fillStyle = "#000"; g.fillText(String(i + 1), x, y);
    });
  }
  function readout(event) {
    if (!detail) return;
    const p = toImage(event.clientX, event.clientY), x = Math.floor(p.x), y = Math.floor(p.y);
    if (x < 0 || y < 0 || x >= detail.width || y >= detail.height) { node("bf-readout").textContent = ""; return; }
    const i = y * detail.width + x, bits = [`x ${x} y ${y}`];
    bits.push({255: "worm", 128: "ignore", 0: "background"}[detail._mask[i]] || "");
    const apValues = trace?.preview?._ap || detail._ap;
    if (apValues && apValues[i]) bits.push(`${trace?.preview ? "preview " : ""}A-P ${((apValues[i] - 1) / 254).toFixed(3)}`);
    if (detail._overlap && detail._overlap[i]) bits.push("overlap");
    if (context && mode === "difference") { const L = maxLag(); bits.push(`Δ ${context.gray[L + lag][i] - context.gray[L - lag][i]}`); }
    node("bf-readout").textContent = bits.filter(Boolean).join(" · ");
  }

  // ------------------------------------------------------------ wiring

  function renderLayers() {
    const box = node("bf-layers"); box.replaceChildren();
    for (const l of layers) {
      const row = document.createElement("label"); row.className = "bf-layer"; row.dataset.bfLayer = l.id;
      row.innerHTML = `<input type="checkbox" ${l.on ? "checked" : ""}><span><span class="swatch" style="background:${l.swatch}"></span>${escapeHtml(l.name)}</span><input type="range" min="0" max="1" step="0.05" value="${l.alpha}" aria-label="${escapeHtml(l.name)} opacity">`;
      row.querySelector("input[type=checkbox]").onchange = e => { l.on = e.target.checked; draw(); };
      row.querySelector("input[type=range]").oninput = e => { l.alpha = Number(e.target.value); draw(); };
      box.append(row);
    }
  }
  function keydown(event) {
    if (state.screen !== "bodyfields" || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || document.querySelector("dialog[open]")) return;
    const target = event.target instanceof Element ? event.target : null;
    if (target && target.closest("input,select,textarea")) return;
    if (target && target.closest("button,a") && [" ", "Enter"].includes(event.key)) return;
    const key = event.key === " " ? "space" : event.key.toLowerCase();
    const viewKeys = {d: () => context && setMode(mode === "frame" ? "difference" : "frame"), 0: () => { fitView(); draw(); }};
    // While tracing only the trace keys and the view keys act, so a stray key cannot leave the sample.
    const tracing = {t: toggleTrace, escape: () => { setTrace(null); status(""); }, backspace: removePoint, enter: () => fitTrace(), ...viewKeys};
    const actions = trace ? tracing : {
      t: toggleTrace,
      n: () => step(1), p: () => step(-1), h: () => !node("bf-flip").disabled && flip(),
      a: () => !node("bf-review-accepted").disabled && review("accepted"), r: () => !node("bf-review-rejected").disabled && review("rejected"),
      ...viewKeys, space: togglePlay,
      arrowleft: () => mode === "frame" ? setOffset(offset - 1) : setLag(lag - 1), arrowright: () => mode === "frame" ? setOffset(offset + 1) : setLag(lag + 1),
    };
    if (!actions[key]) return;
    event.preventDefault();
    actions[key]();
  }
  function init() {
    renderLayers();
    for (const n of ["bf-split", "bf-orientation", "bf-method", "bf-review", "bf-status", "bf-contact"]) node(n).onchange = refresh;
    for (const n of ["bf-min-iou", "bf-max-iou"]) node(n).onchange = refresh;
    node("bf-refresh").onclick = refresh;
    node("bf-prev").onclick = () => step(-1); node("bf-next").onclick = () => step(1);
    node("bf-flip").onclick = flip; node("bf-rebuild").onclick = rebuild; node("bf-edit-mask").onclick = editMask;
    node("bf-trace").onclick = toggleTrace; node("bf-trace-fit").onclick = () => fitTrace(); node("bf-trace-drawn").onclick = () => fitTrace(true);
    node("bf-trace-undo").onclick = removePoint; node("bf-trace-cancel").onclick = () => { setTrace(null); status(""); };
    node("bf-trace-accept").onclick = acceptTrace; node("bf-trace-discard").onclick = discardPreview;
    for (const button of document.querySelectorAll("[data-bf-review]")) button.onclick = () => review(button.dataset.bfReview);
    for (const input of document.querySelectorAll("input[name=bf-mode]")) input.onchange = () => setMode(input.value);
    node("bf-offset").oninput = e => setOffset(Number(e.target.value));
    node("bf-lag").oninput = e => setLag(Number(e.target.value));
    node("bf-gain").onchange = e => { gain = Number(e.target.value); renderBase(); draw(); };
    node("bf-play").onclick = togglePlay;
    node("bf-fps").onchange = () => { if (playTimer) { stopPlay(); togglePlay(); } };
    node("bf-jobs").onclick = () => { if (typeof openRightTab === "function") { $("#jobs-mine").checked = false; openRightTab("jobs"); } };
    const c = canvas();
    c.addEventListener("wheel", event => {
      event.preventDefault();
      const before = toImage(event.clientX, event.clientY), rect = c.getBoundingClientRect();
      view.scale = Math.max(0.05, Math.min(40, view.scale * (event.deltaY > 0 ? 0.9 : 1.1)));
      view.tx = event.clientX - rect.left - before.x * view.scale; view.ty = event.clientY - rect.top - before.y * view.scale;
      draw();
    }, {passive: false});
    c.addEventListener("pointerdown", event => { drag = {x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, button: event.button}; c.setPointerCapture(event.pointerId); });
    c.addEventListener("pointermove", event => {
      if (drag) { view.tx += event.clientX - drag.x; view.ty += event.clientY - drag.y; drag = {x: event.clientX, y: event.clientY}; draw(); }
      else readout(event);
    });
    // In trace mode a left click that did not pan adds a point; dragging still pans.
    c.addEventListener("pointerup", event => {
      if (trace && drag && drag.button === 0 && !busy && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 4) addPoint(toImage(event.clientX, event.clientY));
      drag = null;
    });
    for (const name of ["pointercancel", "lostpointercapture"]) c.addEventListener(name, () => { drag = null; });
    c.addEventListener("dblclick", () => { if (!trace) { fitView(); draw(); } });
    new ResizeObserver(resize).observe(c);
    window.addEventListener("keydown", keydown);
    window.addEventListener("workflow:jobs", onJobs);
    window.addEventListener("workflow:task", event => { if (event.detail.screen !== "bodyfields") stopPlay(); });
    renderSummary(); syncContextControls();
  }
  // Entering the screen re-reads the list and the open sample (a label saved in Paint makes it stale).
  async function show() {
    await refresh();
    if (detail) await open(id(), {keepView: true});
    else if (rows.length) await open(rows[0].sample_id);
  }
  return {init, show, refresh, open, flip, review, rebuild, step, setMode, setOffset, setLag, toggleTrace, fitTrace, acceptTrace,
    trace: () => trace, current: () => detail, layers};
})();
