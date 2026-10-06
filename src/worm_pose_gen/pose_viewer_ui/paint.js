"use strict";

// Paint: corpus labeling, independent of workspaces. The screen first lists
// label groups (/api/labeling/groups: manifests, recording sections chosen for
// relabeling, saved labels); opening one walks its entries in order, starting
// at the first unlabeled one. The frame, proposals and draft belong to the
// entry; saves go to the corpus with the group's split pledges.
// PNG labels use 0 background / 127 ignore / 255 worm.
const paintScreen = (() => {
  let group = null, index = -1, data = null, pixels = null, saved = null, history = [], dirty = false, busy = false;
  let brush = 255, diameter = 12, painting = false, last = null, cursor = null, opacity = 0.45, showRaw = false;
  let proposal = null, proposalName = null, probability = null, request = 0, proposalRequest = 0, refineRequest = 0, refining = false, draftGeneration = 0;
  let view = {scale: 1, tx: 0, ty: 0}, pan = null, returnTo = null, groupsPayload = {groups: [], discovered: []};
  const overlay = document.createElement("canvas"), preview = document.createElement("canvas");
  const node = id => document.getElementById(id);
  const text = (id, message) => { if (node(id)) node(id).textContent = message; };
  const canvas = () => node("paint-canvas");
  const entry = () => group?.entries[index] || null;
  const ready = () => !!pixels && !busy;
  const status = (message, kind) => { const n = node("paint-status"); if (n) { n.textContent = message; n.className = "note" + (kind ? " " + kind : ""); } };
  const isOpen = () => state.screen === "paint" && !!group;

  // ------------------------------------------------------------ groups

  function progressText(p) { return p ? `${p.labeled} labeled · ${p.remaining} remaining · ${p.total} total` : ""; }
  function groupKind(g) { return {manifest: "Manifest", section: "Recording section", samples: "Saved labels"}[g.kind] || g.kind; }
  function groupRow(g, actions) {
    const item = document.createElement("div"); item.className = "item paint-group"; item.dataset.group = g.id || g.path;
    const label = document.createElement("span"); label.textContent = g.name;
    const meta = document.createElement("div"); meta.className = "meta";
    meta.textContent = [groupKind(g), g.progress ? progressText(g.progress) : g.frames !== undefined ? `${g.frames} frames` : "", g.errors?.length ? `${g.errors.length} unavailable recording(s)` : "", g.error || ""].filter(Boolean).join(" · ");
    if (g.description) meta.title = g.description;
    label.append(meta);
    const buttons = document.createElement("span");
    for (const [name, run] of actions) { const b = document.createElement("button"); b.type = "button"; b.textContent = name; b.onclick = event => { event.stopPropagation(); run(); }; buttons.append(b); }
    item.append(label, buttons); item.onclick = () => actions[0][1]();
    return item;
  }
  async function refreshGroups() {
    try { groupsPayload = await api("/api/labeling/groups"); }
    catch (error) { text("paint-groups-status", serverIsApp() ? error.message : "Paint needs the app server."); return; }
    const opened = node("paint-groups"), discovered = node("paint-manifests");
    opened.replaceChildren(...groupsPayload.groups.map(g => groupRow(g, [["Open", () => openGroup(g.id)], [g.kind === "section" ? "Remove" : "Close", () => closeGroup(g)]])));
    if (!groupsPayload.groups.length) opened.innerHTML = '<div class="empty">No label groups open yet.</div>';
    discovered.replaceChildren(...groupsPayload.discovered.map(m => groupRow(m, [["Open", () => create({kind: "manifest", path: m.path})]])));
    if (!groupsPayload.discovered.length) discovered.innerHTML = '<div class="empty">Every repository manifest is open.</div>';
    text("paint-groups-status", "");
    renderRecordings();
  }
  function renderRecordings() {
    const select = node("paint-section-recording"), previous = select.value;
    select.replaceChildren(new Option("Choose a recording", ""));
    for (const rec of state.recordings || []) {
      const option = new Option(`${rec.path} · ${rec.dataset || "/img_nir"}${rec.frames ? ` · ${rec.frames} frames` : ""}`, JSON.stringify([rec.path, rec.dataset || "/img_nir"]));
      option.disabled = rec.readable === false; select.add(option);
    }
    if ([...select.options].some(o => o.value === previous)) select.value = previous;
  }
  async function create(payload, options = {}) {
    try { return await openDetail(await post("/api/labeling/groups", payload), options); }
    catch (error) { text("paint-groups-status", error.message); setStatus(error.message, "error"); return false; }
  }
  async function closeGroup(g) {
    if (g.kind === "section" && !window.confirm(`Remove the section ${g.name}? Saved labels stay in the corpus.`)) return;
    try { await api(`/api/labeling/groups/${encodeURIComponent(g.id)}`, {method: "DELETE"}); await refreshGroups(); }
    catch (error) { text("paint-groups-status", error.message); }
  }
  async function openGroup(id, options = {}) {
    try { return await openDetail(await api(`/api/labeling/groups/${encodeURIComponent(id)}`), options); }
    catch (error) { text("paint-groups-status", error.message); return false; }
  }
  async function openDetail(detail, {position = null, back = null} = {}) {
    if (!await requestLeave()) return false;
    group = detail; returnTo = back; index = -1;
    if (state.screen !== "paint") showTab("paint");
    renderMode();
    const start = position ?? detail.first_unlabeled ?? 0;
    canvas().focus({preventScroll: true});
    if (!detail.entries.length) { status("This group has no entries."); return true; }
    if (detail.first_unlabeled === null && position === null && detail.kind !== "samples") status("Every entry of this group is labeled; showing the first.");
    return go(start);
  }
  // Saved labels from elsewhere in the app (Labels, Body fields); ``back`` names the screen to return to.
  function openSamples(sampleIds, {name = "", back = null} = {}) { return create({kind: "samples", sample_ids: sampleIds, name}, {back}); }
  async function openSection(section) { return create({kind: "section", ...section}); }
  async function chooser() { if (!await requestLeave()) return false; group = null; data = null; pixels = null; renderMode(); await refreshGroups(); return true; }
  async function goBack() {
    if (!returnTo) return chooser();
    const screen = returnTo;
    if (!await requestLeave()) return false;
    group = null; data = null; pixels = null; returnTo = null; renderMode(); showTab(screen); return true;
  }
  async function refreshProgress() {
    if (!group) return;
    const detail = await api(`/api/labeling/groups/${encodeURIComponent(group.id)}`);
    if (detail.id === group.id) group = detail;
    renderQueue();
  }

  // ------------------------------------------------------------ entries

  async function requestLeave() {
    if (busy) { status("Wait for the current save to finish."); return false; }
    if (!dirty) return true;
    const choice = await requestDraftDecision();
    if (choice === "save") return (await save()).ok;
    if (choice === "discard") { discardDraft(); return true; }
    return false;
  }
  async function go(position) {
    if (!group || position < 0 || position >= group.entries.length) return false;
    if (position !== index && !await requestLeave()) return false;
    const wanted = group.entries[position], token = ++request;
    index = position; data = null; pixels = null; proposal = proposalName = probability = null; renderQueue(); draw();
    if (wanted.error) { status(`Unavailable: ${wanted.error}`, "error"); controls(); return false; }
    status("Loading frame…");
    try {
      const payload = await post("/api/labeling/frame", {target: wanted.target, group_id: group.id});
      const [mask, image, raw] = await Promise.all([decodeGray(payload.mask), loadImage(payload.image), loadImage(payload.image_raw)]);
      if (token !== request) return false;
      data = {...payload, _image: image, _raw: raw};
      pixels = maskTools.normalize(mask); saved = pixels.slice(); history = []; dirty = false; draftGeneration++;
      if (!view.fitted || view.size !== `${data.width}x${data.height}`) fitView();
      refreshOverlay(); draftStatus(); renderQueue();
      return true;
    } catch (error) { if (token === request) { status(error.message, "error"); controls(); } return false; }
  }
  const next = () => go(index + 1);
  const previous = () => go(index - 1);
  function nextUnlabeled() {
    const after = group?.entries.findIndex((e, i) => i > index && !e.labeled && !e.error) ?? -1;
    if (after >= 0) return go(after);
    const before = group?.entries.findIndex(e => !e.labeled && !e.error) ?? -1;
    status(before >= 0 ? `No unlabeled entry after this one; entry ${before + 1} is still unlabeled.` : "Every entry of this group is labeled.");
    return Promise.resolve(false);
  }
  function draftStatus() {
    if (!data) return;
    status(`${dirty ? "Unsaved draft" : data.sample ? `Saved label · revision ${data.sample.revision}` : "No saved label yet"}${data.pledged_split ? ` · pledged ${data.pledged_split}` : ""}`);
  }

  // ------------------------------------------------------------ drafts

  function refreshOverlay() {
    if (!data || !pixels) return;
    maskTools.render(overlay, data.width, data.height, pixels);
    if (proposal) maskTools.render(preview, data.width, data.height, proposal, true);
    controls(); draw();
  }
  function pushUndo() { history.push(pixels.slice()); if (history.length > 30) history.shift(); }
  function mutated() { draftGeneration++; dirty = pixels.some((v, i) => v !== saved[i]); refreshOverlay(); draftStatus(); window.dispatchEvent(new CustomEvent("workflow:mask-state")); }
  function undoStroke() { if (!ready() || !history.length) return false; pixels = history.pop(); mutated(); return true; }
  function discardDraft() { if (!pixels || busy) return false; pixels = saved.slice(); history = []; dirty = false; draftGeneration++; refreshOverlay(); draftStatus(); return true; }
  function clearDraft() { if (!ready()) return false; pushUndo(); pixels.fill(0); mutated(); return true; }
  async function save() {
    if (!ready()) return {ok: false};
    busy = true; painting = false; controls();
    const current = entry();
    try {
      const result = await post("/api/labeling/save", {target: data.target, mask: maskTools.encode(data.width, data.height, pixels), revision: data.corpus_revision || 0, group_id: group.id});
      data.sample = result.sample; data.corpus_revision = result.sample.revision; if (data.capabilities) data.capabilities.saved_corpus = true; saved = pixels.slice(); dirty = false; history = [];
      current.labeled = true;
      await refreshProgress().catch(() => {});
      status(`Saved to the corpus (${result.sample.split}, revision ${result.sample.revision}).`, "ok");
      if (typeof corpusUI !== "undefined") corpusUI.refresh().catch(() => {});
      return {ok: true};
    } catch (error) { status(`Not saved: ${error.message}`, "error"); return {ok: false}; }
    finally { busy = false; controls(); }
  }
  async function saveNext() { const result = await save(); if (!result.ok) return false; return group.kind === "samples" ? next() : nextUnlabeled(); }
  function sourceName(name) { return name === "saved" ? "saved_corpus" : name; }
  function updateThreshold() {
    const threshold = Number(node("paint-threshold").value); text("paint-threshold-value", threshold.toFixed(2));
    if (probability && proposalName === "network") { proposal = probability.map(v => v >= threshold * 255 ? 255 : 0); refreshOverlay(); }
  }
  async function selectProposal(name) {
    if (!ready()) return false;
    const source = sourceName(name);
    if (data.capabilities?.[source] === false) { text("paint-proposal-status", "This proposal is unavailable for this frame."); return false; }
    const token = request, ask = ++proposalRequest; proposalName = name; proposal = probability = null; controls(); text("paint-proposal-status", `Loading ${source.replaceAll("_", " ")} preview…`);
    try {
      const response = await post("/api/labeling/proposals", {target: data.target, source, request_id: ask});
      const [mask, prob] = await Promise.all([decodeGray(response.mask), decodeGray(response.probability)]);
      if (token !== request || ask !== proposalRequest) return false;
      if ((mask || prob).length !== pixels.length) throw new Error("Proposal dimensions do not match this frame.");
      probability = prob; proposal = maskTools.normalize(mask);
      if (probability) updateThreshold(); else refreshOverlay();
      text("paint-proposal-status", `${source.replaceAll("_", " ")} preview · Apply changes the draft.`); return true;
    } catch (error) { if (token === request && ask === proposalRequest) { proposal = probability = null; text("paint-proposal-status", error.message); controls(); } return false; }
  }
  function applyProposal(event) {
    if (!ready() || !proposal) return false;
    pushUndo(); pixels = maskTools.combine(pixels, proposal, maskTools.combineMode(event, node("paint-combine").value)); mutated(); return true;
  }
  async function refine(method) {
    if (!ready() || refining) return false;
    const token = request, generation = draftGeneration, ask = ++refineRequest; refining = true; controls(); text("paint-refine-status", `Running ${method.replaceAll("_", " ")}…`);
    try {
      const response = await post("/api/labeling/refine", {target: data.target, mask: maskTools.encode(data.width, data.height, pixels), method, request_id: ask, draft_generation: generation});
      const labels = maskTools.normalize(await decodeGray(response.mask));
      if (token !== request || ask !== refineRequest) return false;
      if (generation !== draftGeneration) { text("paint-refine-status", "The draft changed while refinement ran; its result was discarded."); return false; }
      pushUndo(); pixels = labels; mutated(); text("paint-refine-status", `${method.replaceAll("_", " ")} applied`); return true;
    } catch (error) { if (token === request) text("paint-refine-status", error.message); return false; }
    finally { if (ask === refineRequest) { refining = false; controls(); } }
  }
  function setBrush(value) { if (!ready()) return false; brush = value; controls(); draw(); return true; }
  function resizeBrush(delta) { if (!ready()) return false; diameter = Math.max(1, Math.min(100, diameter + delta)); node("paint-size").value = diameter; text("paint-size-value", diameter + " px"); draw(); return true; }
  function cycleOpacity() { opacity = opacity > 0.4 ? 0.2 : opacity > 0 ? 0 : 0.45; node("paint-opacity").value = String(Math.round(opacity * 100)); draw(); return opacity; }
  function toggleRaw() { showRaw = !showRaw; node("paint-raw").classList.toggle("active", showRaw); node("paint-raw").setAttribute("aria-pressed", String(showRaw)); draw(); }

  // ------------------------------------------------------------ rendering

  function renderMode() {
    node("paint-chooser").hidden = !!group;
    node("paint-workbench").hidden = !group;
    if (group) renderQueue();
    controls();
  }
  function renderQueue() {
    if (!group) return;
    const e = entry(), target = e?.target;
    text("paint-group-name", `${groupKind(group)} · ${group.name}`);
    text("paint-position", e ? `Entry ${index + 1} of ${group.entries.length}` : `${group.entries.length} entries`);
    text("paint-progress", progressText(group.progress));
    const bar = node("paint-progress-bar"); bar.max = group.progress.total || 1; bar.value = group.progress.labeled;
    const where = target ? `${(target.recording || "").split("/").pop()}${target.dataset && target.dataset !== "/img_nir" ? " · " + target.dataset : ""} · frame ${target.frame}` : "";
    text("paint-entry", [where, e?.labeled ? "labeled" : e ? "unlabeled" : "", e && e.split !== "auto" ? `pledged ${e.split}` : "", e?.reasons?.length ? e.reasons.join(", ") : ""].filter(Boolean).join(" · "));
    node("paint-back").textContent = returnTo === "bodyfields" ? "Return to Body fields" : returnTo === "labels" ? "Return to Labels" : "Label groups";
  }
  function controls() {
    const ok = ready();
    for (const id of ["paint-save", "paint-save-next", "paint-apply", "paint-undo", "paint-clear", "paint-discard", "paint-size", "paint-threshold"]) if (node(id)) node(id).disabled = !ok;
    node("paint-undo").disabled = !ok || !history.length;
    node("paint-discard").disabled = !ok || !dirty;
    node("paint-apply").disabled = !ok || !proposal;
    node("paint-threshold").disabled = !ok || data?.capabilities?.network === false;
    node("paint-prev").disabled = busy || !group || index <= 0;
    node("paint-next").disabled = busy || !group || index >= group.entries.length - 1;
    node("paint-next-unlabeled").disabled = busy || !group;
    for (const b of document.querySelectorAll("[data-paint-brush]")) { b.disabled = !ok; b.classList.toggle("active", Number(b.dataset.paintBrush) === brush); }
    for (const b of document.querySelectorAll("[data-paint-proposal]")) { b.disabled = !ok || data?.capabilities?.[sourceName(b.dataset.paintProposal)] === false; b.classList.toggle("active", b.dataset.paintProposal === proposalName); }
    for (const b of document.querySelectorAll("[data-paint-refine]")) b.disabled = !ok || refining;
    if (group) node("paint-save").textContent = "Save label · S";
  }
  function fitView() {
    const c = canvas(); if (!c || !data) return;
    const rect = c.getBoundingClientRect(), scale = Math.min(rect.width / data.width, rect.height / data.height) * 0.95 || 1;
    view = {scale, tx: (rect.width - data.width * scale) / 2, ty: (rect.height - data.height * scale) / 2, fitted: rect.width > 0, size: `${data.width}x${data.height}`};
  }
  function toImage(clientX, clientY) {
    const rect = canvas().getBoundingClientRect();
    return {x: (clientX - rect.left - view.tx) / view.scale, y: (clientY - rect.top - view.ty) / view.scale};
  }
  function resize() {
    const c = canvas(), ratio = window.devicePixelRatio || 1, rect = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(rect.width * ratio)); c.height = Math.max(1, Math.round(rect.height * ratio));
    if (data && !view.fitted) fitView();
    draw();
  }
  function draw() {
    const c = canvas(); if (!c) return;
    const g = c.getContext("2d"), ratio = window.devicePixelRatio || 1;
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height);
    if (!data) return;
    g.setTransform(ratio * view.scale, 0, 0, ratio * view.scale, ratio * view.tx, ratio * view.ty);
    g.imageSmoothingEnabled = false;
    g.drawImage(showRaw && data._raw ? data._raw : data._image, 0, 0);
    g.globalAlpha = opacity;
    if (node("paint-visible").checked) g.drawImage(overlay, 0, 0);
    if (proposal && node("paint-preview").checked) g.drawImage(preview, 0, 0);
    g.globalAlpha = 1;
    if (cursor && pixels) { g.beginPath(); g.arc(cursor.x, cursor.y, diameter / 2, 0, Math.PI * 2); g.strokeStyle = maskTools.brushColor(brush); g.lineWidth = 1.5 / view.scale; g.stroke(); }
  }

  // ------------------------------------------------------------ wiring

  function pointerDown(event) {
    const c = canvas(); c.focus({preventScroll: true});
    if (event.button === 0 && !event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey && ready()) {
      painting = true; last = toImage(event.clientX, event.clientY); cursor = last; pushUndo();
      maskTools.stroke(pixels, data.width, data.height, last, last, diameter, brush); mutated();
    } else pan = {x: event.clientX, y: event.clientY};
    c.setPointerCapture(event.pointerId); event.preventDefault();
  }
  function pointerMove(event) {
    const point = toImage(event.clientX, event.clientY);
    if (painting) { maskTools.stroke(pixels, data.width, data.height, last, point, diameter, brush); last = point; cursor = point; mutated(); return; }
    if (pan) { view.tx += event.clientX - pan.x; view.ty += event.clientY - pan.y; pan = {x: event.clientX, y: event.clientY}; }
    cursor = point; draw();
  }
  function pointerUp() { if (painting) { painting = false; last = null; controls(); } pan = null; }
  function init() {
    node("paint-refresh").onclick = refreshGroups;
    node("paint-manifest-open").onclick = () => { const path = node("paint-manifest-path").value.trim(); if (path) create({kind: "manifest", path}); };
    node("paint-manifest-path").addEventListener("keydown", event => { if (event.key === "Enter") node("paint-manifest-open").click(); });
    node("paint-section-create").onclick = () => {
      const choice = node("paint-section-recording").value;
      if (!choice) { text("paint-groups-status", "Choose a recording for the section."); return; }
      const [recording, dataset] = JSON.parse(choice);
      openSection({recording, dataset, first: Number(node("paint-section-first").value), last: Number(node("paint-section-last").value), step: Number(node("paint-section-step").value || 1), name: node("paint-section-name").value.trim()});
    };
    node("paint-back").onclick = goBack;
    node("paint-prev").onclick = previous; node("paint-next").onclick = next; node("paint-next-unlabeled").onclick = nextUnlabeled;
    node("paint-save").onclick = save; node("paint-save-next").onclick = saveNext;
    node("paint-undo").onclick = undoStroke; node("paint-clear").onclick = clearDraft; node("paint-discard").onclick = discardDraft;
    node("paint-apply").onclick = applyProposal;
    node("paint-raw").onclick = toggleRaw; node("paint-fit").onclick = () => { fitView(); draw(); };
    for (const b of document.querySelectorAll("[data-paint-brush]")) b.onclick = () => setBrush(Number(b.dataset.paintBrush));
    for (const b of document.querySelectorAll("[data-paint-proposal]")) b.onclick = async event => { if (await selectProposal(b.dataset.paintProposal) && (event.shiftKey || event.altKey || event.ctrlKey || event.metaKey)) applyProposal(event); };
    for (const b of document.querySelectorAll("[data-paint-refine]")) b.onclick = () => refine(b.dataset.paintRefine);
    node("paint-size").oninput = event => { diameter = Number(event.target.value); text("paint-size-value", diameter + " px"); draw(); };
    node("paint-threshold").oninput = updateThreshold;
    node("paint-opacity").onchange = event => { opacity = Number(event.target.value) / 100; draw(); };
    for (const id of ["paint-visible", "paint-preview"]) node(id).onchange = draw;
    const c = canvas();
    c.addEventListener("pointerdown", pointerDown);
    c.addEventListener("pointermove", pointerMove);
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) c.addEventListener(name, pointerUp);
    c.addEventListener("pointerleave", () => { cursor = null; draw(); });
    c.addEventListener("contextmenu", event => event.preventDefault());
    c.addEventListener("wheel", event => {
      event.preventDefault();
      const before = toImage(event.clientX, event.clientY), rect = c.getBoundingClientRect();
      view.scale = Math.max(0.05, Math.min(40, view.scale * (event.deltaY > 0 ? 0.9 : 1.1)));
      view.tx = event.clientX - rect.left - before.x * view.scale; view.ty = event.clientY - rect.top - before.y * view.scale;
      draw();
    }, {passive: false});
    new ResizeObserver(resize).observe(c);
    window.addEventListener("beforeunload", event => { if (dirty || busy) { event.preventDefault(); event.returnValue = ""; } });
    renderMode();
  }
  // Entering the screen without a group shows the chooser; a launcher-opened group (--queue) opens directly.
  async function show() {
    if (!group) await refreshGroups();
    else requestAnimationFrame(resize);
  }
  return {init, show, refreshGroups, openGroup, openSamples, openSection, chooser, goBack, go, next, previous, nextUnlabeled, save, saveNext,
    undoStroke, clearDraft, discardDraft, selectProposal, applyProposal, refine, setBrush, resizeBrush, cycleOpacity, toggleRaw,
    fitView: () => { fitView(); draw(); }, requestLeave, isOpen, isDirty: () => dirty || busy, group: () => group, position: () => index, current: () => data};
})();
