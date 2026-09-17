"use strict";

// Sidebar panels beyond the frame view: the resizable layout, the sidebar
// tabs, the recording browser with workspace creation and run import, the
// pipeline stages with their parameter forms, and the job list.

// ---------------------------------------------------------------- layout
//
// The three panels around the stage are sized by CSS variables; the
// splitters drag them, their buttons collapse and restore them, and the
// layout persists in localStorage.

const LAYOUT_DEFAULT = { left: 320, right: 340, bottom: 150 };
const LAYOUT_MIN = { left: 280, right: 220, bottom: 100 };
const layout = { sizes: { ...LAYOUT_DEFAULT }, collapsed: { right: false }, restore: {} };

function readStorage(key, fallback) {
  try { const raw = localStorage.getItem(key); return raw === null ? fallback : JSON.parse(raw); } catch (error) { return fallback; }
}

function writeStorage(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (error) { /* blocked storage */ }
}

function loadLayout() {
  const saved = readStorage("poseApp.taskLayout", null);
  if (saved && saved.sizes) { Object.assign(layout.sizes, saved.sizes); Object.assign(layout.collapsed, saved.collapsed || {}); }
}

function saveLayout() {
  writeStorage("poseApp.taskLayout", { sizes: layout.sizes, collapsed: layout.collapsed });
}

function applyLayout() {
  const app = $("#app");
  for (const side of ["left", "right", "bottom"]) {
    const size = layout.collapsed[side] ? (side === "bottom" ? 65 : 0) : layout.sizes[side];
    app.style.setProperty(`--${side}`, `${size}px`);
    const splitter = $(`#split-${side}`);
    splitter.classList.toggle("collapsed", !!layout.collapsed[side]);
    const button = splitter.querySelector("button");
    button.textContent = { left: layout.collapsed.left ? "▸" : "◂", right: layout.collapsed.right ? "◂" : "▸", bottom: layout.collapsed.bottom ? "▴" : "▾" }[side];
    button.setAttribute("aria-expanded", String(!layout.collapsed[side]));
    button.title = layout.collapsed[side] ? "expand panel" : "collapse panel";
  }
  $("#timeline").classList.toggle("collapsed", !!layout.collapsed.bottom);
  $("#details").hidden = !!layout.collapsed.right;
  app.dataset.rightOpen = String(!layout.collapsed.right);
  syncInspectorToggle();
}

function syncInspectorToggle() {
  for (const id of ["toggle-inspector", "review-frame-tools"]) {
    const active = !layout.collapsed.right && (id === "toggle-inspector" || (state.screen === "workspace" && (state.rightTab || "statistics") === "statistics"));
    const button = $(`#${id}`);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function togglePanel(side) {
  layout.collapsed[side] = !layout.collapsed[side];
  applyLayout();
  saveLayout();
}

function initSplitters() {
  loadLayout();
  applyLayout();
  for (const side of ["left", "right", "bottom"]) {
    const splitter = $(`#split-${side}`);
    splitter.querySelector("button").addEventListener("click", (event) => { event.stopPropagation(); togglePanel(side); });
    splitter.addEventListener("dblclick", () => { layout.sizes[side] = LAYOUT_DEFAULT[side]; layout.collapsed[side] = false; applyLayout(); saveLayout(); });
    splitter.addEventListener("pointerdown", (event) => {
      if (event.target.tagName === "BUTTON") return;
      event.preventDefault();
      splitter.setPointerCapture(event.pointerId);
      const rect = $("#app").getBoundingClientRect();
      const move = (e) => {
        let size = side === "left" ? e.clientX - rect.left : side === "right" ? rect.right - e.clientX : rect.bottom - e.clientY;
        size = Math.round(Math.max(0, Math.min(size, side === "bottom" ? rect.height * 0.8 : rect.width * 0.5)));
        if (size < LAYOUT_MIN[side] / 2) { layout.collapsed[side] = true; }
        else { layout.collapsed[side] = false; layout.sizes[side] = Math.max(LAYOUT_MIN[side], size); }
        applyLayout();
      };
      const up = () => { splitter.removeEventListener("pointermove", move); splitter.removeEventListener("pointerup", up); saveLayout(); };
      splitter.addEventListener("pointermove", move);
      splitter.addEventListener("pointerup", up);
    });
  }
}

// ---------------------------------------------------------------- tabs

const TABS = ["rerun", "inspect", "paint", "review", "export"];
const TAB_ALIASES = { view: "inspect", data: "open", pipeline: "rerun", run: "rerun", compare: "review", corpus: "labels" };

const taskScrollPositions = new Map();
function showTab(name) {
  const previous = currentTab();
  if (state.screen === "workspace") taskScrollPositions.set(previous, $("#sidebar").scrollTop);
  name = TAB_ALIASES[name] || name;
  if (["import", "open"].includes(name) && !["import", "open"].includes(previous)) state.panelReturn = previous;
  const screen = ["import", "open", "labels", "training"].includes(name) ? name : "workspace";
  if (state.run && previous !== name) state.userView = true;
  if (screen === "workspace" && !TABS.includes(name)) name = "inspect";
  if (screen === "workspace") state.activeTask = name;
  closeDrawer();
  state.screen = screen;
  $("#app").dataset.screen = screen;
  syncInspectorToggle();
  const previewProgress = $("#preview-progress");
  if (previewProgress) (screen === "import" ? $("#import-preview-progress") : document.body).append(previewProgress);
  if (screen === "open") renderOpenSelection();
  for (const button of document.querySelectorAll('.file-actions [data-screen]')) {
    const active = button.dataset.screen === screen;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  $("#app").dataset.task = state.activeTask || "inspect";
  for (const tab of TABS) {
    $(`#tab-${tab}`).hidden = tab !== state.activeTask;
    const button = $(`#tabs button[data-tab="${tab}"]`);
    button.classList.toggle("active", tab === state.activeTask);
    button.setAttribute("aria-pressed", String(tab === state.activeTask));
  }
  for (const button of document.querySelectorAll("[data-main-tab]")) {
    const active = button.dataset.mainTab === screen;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  }
  for (const id of ["import", "open", "labels", "training"]) $(`#screen-${id}`).hidden = screen !== id;
  const painting = screen === "workspace" && name === "paint";
  $("#paint-tray").hidden = !painting;
  $("#paint-display").hidden = !painting;
  if (painting && state.playing) togglePlay();
  if (typeof maskEditor !== "undefined" && maskEditor.setActive) maskEditor.setActive(painting);
  writeStorage("poseApp.task", state.activeTask || "inspect");
  if ((name === "labels" || name === "training") && typeof corpusUI !== "undefined") corpusUI.refresh();
  if (name === "import") {
    if (!state.recordings.length && serverIsApp()) loadRecordings(false);
    toggleExplorer(true);
  }
  if (typeof syncTaskSelection === "function") syncTaskSelection();
  window.dispatchEvent(new CustomEvent("workflow:task", {detail: {task: name, screen}}));
  if (screen === "workspace" && name !== previous) $("#sidebar").scrollTop = taskScrollPositions.get(name) || 0;
  requestAnimationFrame(() => { resizeCanvas(); drawCharts(); });
}

function currentTab() { return state.screen && state.screen !== "workspace" ? state.screen : state.activeTask || "inspect"; }

function initTabs() {
  for (const button of document.querySelectorAll("button[data-tab]")) button.addEventListener("click", () => showTab(button.dataset.tab));
  showTab(readStorage("poseApp.task", "inspect"));
}

// ---------------------------------------------------------------- catalog

// Re-read the catalog (runs, workspaces, gpus) and redraw what depends on it.
async function refreshCatalog(rescan = false) {
  const info = await api(rescan ? "/api/state?rescan=1" : "/api/state");
  state.info = info;
  renderJobGpuControls();
  if (typeof paintNavigation !== "undefined" && paintNavigation.onCatalog) paintNavigation.onCatalog(info);
  renderServerNotice(info);
  state.runs = info.runs || [];
  state.workspaces = (info.workspaces || []).map((w) => (w.summary ? w : { ...w, summary: w.summary || {} }));
  renderRunList();
  renderOpenSelection();
  renderJobsBadge();
  if (state.recording) renderRecordingDetail();
  return info;
}

// ---------------------------------------------------------------- which server

// The same UI is served by the read-only run viewer (worm-pose-viewer) and by
// the app (worm-pose-app), on the same default port.  Only the app has
// recordings, workspaces and jobs; against the viewer those panels would
// fail with 404s, so say so up front and disable what cannot work.
const VIEWER_ONLY_NOTE = "This server is the read-only run viewer (worm-pose-viewer). Recordings, workspaces, the file explorer and jobs need the app: stop this server and start `worm-pose-app` (python -m worm_pose_gen.app) on this port, then reload.";

function serverIsApp() { return !!(state.info && (state.info.server === "app" || state.info.workspaces !== undefined)); }

function renderServerNotice(info) {
  const isApp = !!(info && (info.server === "app" || info.workspaces !== undefined));
  for (const id of ["server-note-data", "server-note-pipeline"]) {
    const node = $(`#${id}`);
    if (!node) continue;
    node.hidden = isApp;
    node.textContent = isApp ? "" : VIEWER_ONLY_NOTE;
  }
  for (const id of ["explorer-toggle", "recordings-rescan", "import-run", "ws-create"]) {
    const node = $(`#${id}`);
    if (node) { node.disabled = !isApp; node.title = isApp ? node.dataset.title || node.title : "needs worm-pose-app"; }
  }
  if (!isApp) setStatus("connected to the read-only viewer; start worm-pose-app for recordings, workspaces and jobs", "error");
}

// ---------------------------------------------------------------- recordings

let thumbTimer = null;

async function loadRecordings(rescan) {
  $("#recording-count").textContent = "scanning…";
  try {
    const payload = await api(`/api/recordings?rescan=${rescan ? 1 : 0}`);
    state.recordings = Array.isArray(payload) ? payload : payload.recordings || [];
    renderRecordings();
    const roots = (state.info && state.info.recording_roots) || [];
    $("#recording-roots").textContent = roots.length ? `roots: ${roots.join(", ")}` : "";
  } catch (error) {
    $("#recording-count").textContent = "";
    setStatus(error.message, "error");
  }
}

function recordingMatches(rec, filter) {
  if (!filter) return true;
  const haystack = `${rec.name} ${rec.path} ${(rec.runs || []).join(" ")} ${(rec.workspaces || []).join(" ")}`.toLowerCase();
  return haystack.includes(filter);
}

function renderRecordings() {
  const filter = $("#recording-filter").value.trim().toLowerCase();
  const body = $("#recording-table tbody");
  body.innerHTML = "";
  let shown = 0;
  for (const rec of state.recordings) {
    if (!recordingMatches(rec, filter)) continue;
    shown++;
    const tr = document.createElement("tr");
    tr.dataset.path = rec.path;
    if (state.recording && state.recording.path === rec.path) tr.classList.add("current");
    if (!rec.readable) tr.classList.add("unreadable");
    if (rec.registered) tr.classList.add("registered");
    const dims = rec.height && rec.width ? `${rec.height}×${rec.width}` : "";
    tr.title = `${rec.path}${dims ? " · " + dims : ""}${rec.dataset && rec.dataset !== "/img_nir" ? " · dataset " + rec.dataset : ""}${rec.registered ? " · added by hand" : ""}${rec.error ? "\n" + rec.error : ""}`;
    tr.innerHTML = `<td><button type="button" class="recording-select" aria-label="Select recording ${escapeHtml(rec.name)}" aria-pressed="${!!(state.recording && state.recording.path === rec.path)}">${escapeHtml(rec.name)}</button></td><td class="num">${rec.frames === null || rec.frames === undefined ? "–" : rec.frames.toLocaleString()}</td><td class="num">${fmtBytes(rec.size_bytes)}</td>` +
      `<td class="flags"><span class="${rec.readable ? "ok" : "bad"}" title="${rec.readable ? "readable" : escapeHtml(rec.error || "not readable")}">${rec.readable ? "✓" : "✗"}</span>` +
      `<span class="${rec.prior_cached ? "ok" : "dim"}" title="${rec.prior_cached ? "recording prior cached" : "no cached prior"}">P</span></td>` +
      `<td class="num">${(rec.runs || []).length || ""}</td><td class="num">${(rec.workspaces || []).length || ""}</td>`;
    tr.addEventListener("click", () => selectRecording(rec.path));
    tr.addEventListener("pointerenter", () => previewThumbnail(rec));
    body.appendChild(tr);
  }
  $("#recording-table").addEventListener("pointerleave", () => { clearTimeout(thumbTimer); thumbTimer = null; if (state.recording) setThumbnail(state.recording, thumbFrame()); }, { once: true });
  const unreadable = state.recordings.filter((r) => !r.readable).length;
  $("#recording-count").textContent = `${shown}/${state.recordings.length}${unreadable ? ` · ${unreadable} unreadable` : ""}`;
}

function thumbFrame() { return Math.max(0, parseInt($("#thumb-frame").value, 10) || 0); }

function thumbnailUrl(rec, frame) {
  return `/api/recordings/thumbnail?path=${encodeURIComponent(rec.path)}&frame=${frame}&scale=0.25`;
}

function setThumbnail(rec, frame) {
  const img = $("#recording-thumb");
  if (!rec || !rec.readable) { img.removeAttribute("src"); img.alt = rec ? "not readable" : ""; $("#thumb-label").textContent = rec ? `${rec.name}: not readable` : ""; return; }
  const bounded = Math.min(frame, Math.max(0, (rec.frames || 1) - 1));
  img.alt = `${rec.name} frame ${bounded}`;
  $("#thumb-label").textContent = `Loading preview: ${rec.name} · frame ${bounded}…`;
  img.onload = () => { $("#thumb-label").textContent = `${rec.name} · frame ${bounded}`; };
  img.onerror = () => { $("#thumb-label").textContent = `Could not load preview for ${rec.name}`; };
  img.src = thumbnailUrl(rec, bounded);
}

// Hovering a row shows its thumbnail after a short rest; the selection's
// thumbnail comes back when the pointer leaves the table.
function previewThumbnail(rec) {
  clearTimeout(thumbTimer);
  thumbTimer = setTimeout(() => { thumbTimer = null; $("#recording-detail").hidden = false; setThumbnail(rec, thumbFrame()); }, 350);
}

function selectRecording(path) {
  state.recording = state.recordings.find((r) => r.path === path) || null;
  const focusedRecording = document.activeElement?.closest?.("tr[data-path]")?.dataset.path;
  renderRecordings();
  if (focusedRecording) [...document.querySelectorAll("#recording-table tr[data-path]")].find(row => row.dataset.path === focusedRecording)?.querySelector("button")?.focus();
  renderRecordingDetail();
  resetWorkspaceForm();
}

function defaultWorkspaceName(rec, first, last) { return `${rec.name}_f${first}-${last}`; }

function renderRecordingDetail() {
  const rec = state.recording;
  const box = $("#recording-detail");
  $("#import-preview-empty").hidden = !!rec;
  if (!rec) { box.hidden = true; return; }
  box.hidden = false;
  setThumbnail(rec, thumbFrame());
  const dims = rec.height && rec.width ? `${rec.height}×${rec.width}` : "?";
  $("#recording-facts").textContent = [
    rec.path + (rec.dataset && rec.dataset !== "/img_nir" ? `  (dataset ${rec.dataset})` : ""),
    `${rec.frames == null ? "?" : rec.frames.toLocaleString()} frames · ${dims} · ${fmtBytes(rec.size_bytes)} · modified ${fmtTime(rec.modified_at)}`,
    `${rec.readable ? "readable" : "NOT readable: " + (rec.error || "")} · prior ${rec.prior_cached ? "cached" : "not cached"}${rec.registered ? " · added by hand" : ""}`,
  ].join("\n");
  const actions = $("#recording-actions");
  actions.innerHTML = "";
  if (rec.registered) {
    const remove = document.createElement("button");
    remove.type = "button"; remove.textContent = "Remove from recordings"; remove.title = "forget this hand-added recording (the file is not touched)";
    remove.addEventListener("click", async () => {
      try { await post("/api/recordings/unregister", { path: rec.path }); state.recording = null; await loadRecordings(false); renderRecordingDetail(); setStatus(`${rec.name} removed from the recordings`, "ok"); }
      catch (error) { setStatus(error.message, "error"); }
    });
    actions.appendChild(remove);
  }
  // Runs of this recording, each importable as a workspace.
  const runs = $("#recording-runs");
  runs.innerHTML = "";
  const runNames = rec.runs && rec.runs.length ? rec.runs : state.runs.filter((r) => r.recording === rec.name).map((r) => r.name);
  if (!runNames.length) runs.innerHTML = '<div class="empty">no runs of this recording</div>';
  for (const name of runNames) {
    const entry = runEntry(name);
    const item = document.createElement("div");
    item.className = "item";
    item.innerHTML = `<span>${escapeHtml(name)}${entry && entry.frames ? `<div class="meta">frames ${entry.frames[0]}–${entry.frames[1]} · IoU ${fmt(entry.iou_median)}</div>` : ""}</span><span><button type="button" data-open title="open read-only">open</button> <button type="button" data-import title="copy this run into a new workspace">import</button></span>`;
    item.querySelector("[data-open]").addEventListener("click", (e) => { e.stopPropagation(); if (entry) { selectSource("run", name); showTab("view"); } else setStatus("run not in the viewer's catalog; press ↻ next to the filter", "error"); });
    item.querySelector("[data-import]").addEventListener("click", (e) => { e.stopPropagation(); importRun(name); });
    runs.appendChild(item);
  }
  // Workspaces of this recording.
  const wss = $("#recording-workspaces");
  wss.innerHTML = "";
  const wsNames = new Set([...(rec.workspaces || []), ...state.workspaces.filter((w) => recordingStem(w.recording) === rec.name).map((w) => w.name)]);
  if (!wsNames.size) wss.innerHTML = '<div class="empty">no workspaces yet</div>';
  for (const name of wsNames) {
    const entry = workspaceEntry(name);
    const item = document.createElement("div");
    item.className = "item" + (isWorkspace() && state.runName === name ? " current" : "");
    const detail = entry && entry.error ? `<div class="meta">unreadable: ${escapeHtml(entry.error)}</div>` : entry && entry.frames ? `<div class="meta">frames ${entry.frames[0]}–${entry.frames[1]} · ${fmt((entry.summary || {}).fitted)} fitted</div>` : "";
    item.innerHTML = `<span>${escapeHtml(name)}${detail}</span><span><button type="button" ${entry && entry.error ? "disabled" : ""}>open</button></span>`;
    item.addEventListener("click", () => { if (entry && !entry.error) { selectSource("workspace", name); showTab("view"); } else if (!entry) setStatus("workspace not in the catalog yet; press ↻ next to the filter", "error"); });
    wss.appendChild(item);
  }
  $("#ws-create").disabled = !rec.readable;
  if (typeof workflowRun !== "undefined") workflowRun.renderImport();
  $("#ws-form-note").textContent = rec.readable ? "" : "the recording is not readable; a workspace cannot be created";
}

// New workspace form defaults: the entire recording, name from the range.
// Only a new recording selection resets them; a catalog refresh while the
// user types (a job finished elsewhere) leaves the form alone.
function resetWorkspaceForm() {
  const rec = state.recording;
  if (!rec) return;
  const frames = rec.frames || 0;
  const first = $("#ws-first"), last = $("#ws-last");
  first.max = Math.max(0, frames - 1); last.max = Math.max(0, frames - 1);
  first.value = 0;
  last.value = Math.max(0, frames - 1);
  $("#ws-step").value = 1;
  $("#ws-name").dataset.auto = "1";
  $("#ws-name").value = defaultWorkspaceName(rec, first.value, last.value);
  if (typeof workflowRun !== "undefined") workflowRun.resetImport();
}

function syncWorkspaceName() {
  const name = $("#ws-name");
  if (typeof workflowRun !== "undefined") workflowRun.renderImport();
  if (name.dataset.auto !== "1" || !state.recording) return;
  name.value = defaultWorkspaceName(state.recording, $("#ws-first").value, $("#ws-last").value);
}

async function createWorkspace() {
  const rec = state.recording;
  if (!rec) return;
  const first = Number($("#ws-first").value), last = Number($("#ws-last").value), step = Number($("#ws-step").value);
  const name = $("#ws-name").value.trim();
  if (!name) { setStatus("give the workspace a name", "error"); return; }
  if (!(Number.isInteger(first) && Number.isInteger(last) && Number.isInteger(step) && step >= 1 && first >= 0 && last >= first)) { setStatus("first must be ≥ 0 and last ≥ first", "error"); return; }
  if (rec.frames && last >= rec.frames) { setStatus(`last frame must be below ${rec.frames}`, "error"); return; }
  setLoading(1);
  try {
    const info = await post("/api/workspaces", { name, recording: rec.path, first, last, step });
    setStatus(`workspace ${info.name} created (${info.frame_count} frames)`, "ok");
    await refreshCatalog();
    await loadRecordings(false);
    await selectSource("workspace", info.name);
    setRerunScope("workspace");
    showTab("rerun");
  } catch (error) { setStatus(error.message, "error"); } finally { setLoading(-1); }
}

async function importRun(runName) {
  setLoading(1);
  try {
    const info = await post("/api/workspaces/import", { run: runName });
    setStatus(`run imported as workspace ${info.name}`, "ok");
    await refreshCatalog();
    if (state.recordings.length) await loadRecordings(false);
    await selectSource("workspace", info.name);
    showTab("view");
  } catch (error) { setStatus(error.message, "error"); } finally { setLoading(-1); }
}

// ---------------------------------------------------------------- file explorer
//
// Browse the file system for an HDF5 video that is not under a recording
// root, validate its /img_nir video, and register it so it appears
// in the recordings table and can back a workspace.

const explorer = { path: null, file: null, datasets: [], dataset: null, request: 0 };

function toggleExplorer(open) {
  const box = $("#explorer");
  const show = open === undefined ? box.hidden : open;
  box.hidden = !show;
  $("#explorer-toggle").classList.toggle("active", show);
  $("#explorer-toggle").setAttribute("aria-pressed", String(show));
  $("#explorer-toggle").textContent = show ? "Hide explorer" : "Add recording…";
  if (show && explorer.path === null) browseTo(null);
}

function explorerError(message) {
  $("#explorer-entries").replaceChildren();
  $("#explorer-error").textContent = message;
  $("#explorer-error").hidden = false;
  $("#explorer-path").setAttribute("aria-invalid", "true");
  $("#explorer-recovery").hidden = false;
  $("#explorer-retry").onclick = () => browseTo($("#explorer-path").value.trim() || null);
  $("#explorer-choose").onclick = () => { $("#explorer-path").focus(); $("#explorer-path").select(); };
  const attempted = $("#explorer-path").value.trim().replace(/\/+$/, "");
  const parent = attempted.includes("/") ? attempted.slice(0, attempted.lastIndexOf("/")) || "/" : explorer.path;
  $("#explorer-up").disabled = !parent || parent === attempted;
  $("#explorer-up").dataset.parent = parent || "";
  $("#explorer-file").hidden = true;
}

async function browseTo(path) {
  if (!serverIsApp()) { explorerError(VIEWER_ONLY_NOTE); return; }
  const request = ++explorer.request;
  const all = $("#explorer-all").checked ? "&all=1" : "";
  try {
    const listing = await api(`/api/files?${path ? `path=${encodeURIComponent(path)}` : ""}${all}`);
    if (request !== explorer.request) return;
    $("#explorer-error").hidden = true;
    $("#explorer-recovery").hidden = true;
    $("#explorer-path").removeAttribute("aria-invalid");
    explorer.path = listing.path;
    explorer.file = null; explorer.datasets = []; explorer.dataset = null;
    $("#explorer-path").value = listing.path;
    $("#explorer-up").disabled = !listing.parent;
    $("#explorer-up").dataset.parent = listing.parent || "";
    const breadcrumbs = $("#explorer-breadcrumbs");
    breadcrumbs.replaceChildren();
    const parts = listing.path.split("/").filter(Boolean);
    for (let i = 0; i <= parts.length; i++) {
      const target = "/" + parts.slice(0, i).join("/");
      const button = document.createElement("button");
      button.type = "button"; button.textContent = i === 0 ? "/" : parts[i - 1]; button.title = target;
      button.disabled = target === listing.path;
      button.onclick = () => browseTo(target);
      breadcrumbs.append(button);
    }
    const shortcuts = $("#explorer-shortcuts");
    shortcuts.innerHTML = "";
    for (const s of listing.shortcuts || []) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = s.name; b.title = s.path;
      b.addEventListener("click", () => browseTo(s.path));
      shortcuts.appendChild(b);
    }
    const list = $("#explorer-entries");
    list.innerHTML = "";
    if (!listing.entries.length) {
      list.innerHTML = `<div class="empty">no directories${listing.all_files ? " or files" : " or HDF5 files (" + (listing.suffixes || [".h5", ".hdf5"]).join(", ") + "; tick \"show all files\" for others)"} in ${escapeHtml(listing.path)}</div>`;
    }
    const dirs = listing.entries.filter((e) => e.kind === "dir").length;
    for (const entry of listing.entries) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = `item ${entry.kind}${entry.registered ? " registered" : ""}${entry.readable ? "" : " unreadable"}`;
      item.title = entry.path + (entry.readable ? "" : " (not readable)") + (entry.kind === "file" ? " · click to check whether it is an HDF5 file" : "");
      item.innerHTML = `<span>${escapeHtml(entry.name)}</span><span class="meta">${entry.kind === "dir" ? "" : fmtBytes(entry.size_bytes)}</span>`;
      item.addEventListener("click", () => { if (entry.kind === "dir") browseTo(entry.path); else inspectFile(entry.path); });
      list.appendChild(item);
    }
    $("#explorer-file").hidden = true;
    setStatus(`${listing.path}: ${dirs} directories, ${listing.entries.length - dirs} files shown`, "ok");
  } catch (error) { if (request !== explorer.request) return; explorerError(`cannot list ${path || "the default directory"}: ${error.message}`); setStatus(error.message, "error"); }
}

async function inspectFile(path) {
  const request = ++explorer.request;
  const progress = beginPreviewTask("Inspecting video");
  progress.update("datasets", "Reading video frame dimensions…");
  $("#explorer-file").hidden = true;
  try {
    const info = await api(`/api/recordings/datasets?path=${encodeURIComponent(path)}`);
    if (request !== explorer.request) { progress.finish(); return; }
    explorer.file = info.path; explorer.datasets = info.datasets; explorer.dataset = "/img_nir";
    $("#explorer-file").hidden = false;
    $("#explorer-file-name").textContent = `${info.path}${info.registered ? " · already in the recordings" : ""}`;
    renderImportFile();
    progress.finish();
  } catch (error) { if (request === explorer.request) { progress.finish(error); setStatus(error.message, "error"); } else progress.finish(); }
}

function renderImportFile() {
  const video = explorer.datasets.find((dataset) => dataset.name === "/img_nir" && dataset.video);
  $("#explorer-register").disabled = !video;
  $("#explorer-note").textContent = video
    ? `${video.shape[0].toLocaleString()} frames · ${video.shape[2]} × ${video.shape[1]} px`
    : "This file does not contain a video at /img_nir.";
}

async function registerRecording() {
  if (!explorer.file || !explorer.dataset) return;
  const progress = beginPreviewTask("Importing video");
  const source = {path: explorer.file, dataset: explorer.dataset};
  $("#explorer-register").disabled = true;
  setLoading(1);
  try {
    progress.update("register", "Checking video readability and adding it to recordings…");
    const rec = await post("/api/recordings/register", source);
    await prepareRecording(source, progress);
    progress.update("catalog", "Refreshing recording details…");
    await loadRecordings(false);
    progress.update("preview", "Loading corrected preview…");
    selectRecording(rec.path);
    await $("#recording-thumb").decode();
    toggleExplorer(true);
    setStatus(`${rec.name} imported; illumination correction and preview ready`, "ok");
    progress.finish();
  } catch (error) { progress.finish(error); setStatus(error.message, "error"); }
  finally { setLoading(-1); renderImportFile(); }
}

// ---------------------------------------------------------------- stages

// The /api/stages payload in any of its plausible shapes -> [{name, params}].
function normaliseStages(payload) {
  const list = Array.isArray(payload) ? payload : Array.isArray(payload.stages) ? payload.stages : Object.entries(payload.stages || payload).map(([name, value]) => ({ name, ...(Array.isArray(value) ? { params: value } : value) }));
  const stages = list.map((item) => (typeof item === "string" ? { name: item, params: [] } : { name: item.name || item.stage, params: item.params || item.parameters || item.schema || [] }));
  stages.sort((a, b) => (STAGE_ORDER.indexOf(a.name) + 1 || 99) - (STAGE_ORDER.indexOf(b.name) + 1 || 99));
  return stages;
}

async function loadStages() {
  try {
    state.stages = normaliseStages(await api("/api/stages"));
    state.stageValues = readStorage("poseViewer.stageParams", {});
    renderStages();
  } catch (error) {
    state.stagesError = error.message;
    renderStages();
  }
}

function paramInputType(param) {
  const type = String(param.type || "str").toLowerCase();
  if (type === "bool") return "bool";
  if (type === "int") return "int";
  if (type === "float") return "float";
  if (type === "dict" || type.startsWith("list") || type.startsWith("tuple")) return "json";
  return "str";
}

// The edited value of a parameter, or undefined when it is at its default.
function paramValue(stage, param) {
  const values = state.stageValues[stage] || {};
  return param.name in values ? values[param.name] : undefined;
}

function renderParamField(stage, param) {
  const kind = paramInputType(param);
  const edited = paramValue(stage, param);
  const value = edited === undefined ? param.default : edited;
  const field = document.createElement("label");
  field.className = "param" + (edited === undefined ? "" : " edited");
  field.title = `${param.help || ""}\ndefault: ${JSON.stringify(param.default)}`.trim();
  const name = document.createElement("span");
  name.className = "param-name";
  name.textContent = param.name;
  field.appendChild(name);
  let input;
  if (kind === "bool") {
    input = document.createElement("input"); input.type = "checkbox"; input.checked = !!value;
  } else if (kind === "json") {
    input = document.createElement("input"); input.type = "text"; input.value = value === null || value === undefined ? "" : JSON.stringify(value); input.placeholder = "JSON";
  } else {
    input = document.createElement("input"); input.type = kind === "str" ? "text" : "number";
    if (kind === "float") input.step = "any";
    if (kind === "int") input.step = "1";
    input.value = value === null || value === undefined ? "" : String(value);
    input.placeholder = param.default === null || param.default === undefined ? "none" : String(param.default);
  }
  input.addEventListener("change", () => {
    const parsed = parseParamInput(kind, input);
    if (parsed.error) { setStatus(`${stage}.${param.name}: ${parsed.error}`, "error"); return; }
    const values = state.stageValues[stage] || (state.stageValues[stage] = {});
    if (JSON.stringify(parsed.value) === JSON.stringify(param.default)) delete values[param.name]; else values[param.name] = parsed.value;
    writeStorage("poseViewer.stageParams", state.stageValues);
    field.classList.toggle("edited", param.name in values);
    renderStageMeta(stage);
  });
  field.appendChild(input);
  return field;
}

function parseParamInput(kind, input) {
  if (kind === "bool") return { value: input.checked };
  const text = input.value.trim();
  if (kind === "json") {
    if (!text) return { value: null };
    try { return { value: JSON.parse(text) }; } catch (error) { return { error: "not valid JSON" }; }
  }
  if (!text) return { value: null };
  if (kind === "int") { const v = Number(text); return Number.isInteger(v) ? { value: v } : { error: "expected an integer" }; }
  if (kind === "float") { const v = Number(text); return Number.isFinite(v) ? { value: v } : { error: "expected a number" }; }
  return { value: text };
}

// The parameters to send for a stage: the edited values only, so the server's
// defaults apply to everything else.
function collectParams(stage) {
  return { ...(state.stageValues[stage] || {}) };
}

function stageInfo(name) { return (state.stages || []).find((s) => s.name === name); }

function jobStage(job) {
  const spec = job.spec || {};
  const params = spec.params || {};
  if (spec.kind === "region") return `region ${params.algorithm || spec.algorithm || ""}`.trim();
  return spec.stage || params.stage || (spec.label || "").split(/[ :]/)[0] || spec.kind;
}

function jobsOf(stage, workspace) {
  return state.jobs.filter((j) => (j.spec || {}).workspace === workspace && jobStage(j) === stage);
}

function renderStageMeta(stage) {
  const node = $(`.stage[data-stage="${stage}"] .stage-meta`);
  if (!node) return;
  const edited = Object.keys(state.stageValues[stage] || {}).length;
  const parts = [];
  if (edited) parts.push(`${edited} param${edited === 1 ? "" : "s"} changed`);
  const last = jobsOf(stage, state.runName)[0];
  if (last) parts.push(`last: ${last.state}${last.finished_at ? " " + fmtTime(last.finished_at) : last.state === "running" ? ` ${Math.round(100 * (last.progress || 0))}%` : ""}`);
  node.textContent = parts.join(" · ");
  const stageNode = node.closest(".stage");
  stageNode.classList.toggle("running", !!last && (last.state === "running" || last.state === "queued"));
  stageNode.classList.toggle("failed", !!last && last.state === "failed");
  stageNode.classList.toggle("done", !!last && last.state === "done");
}

// The stage forms are built once per stage list and updated in place after
// that (target, Run buttons, meta lines), so a refresh while the user edits a
// parameter does not throw the edit away.
function renderStages() {
  const box = $("#stages");
  const workspace = isWorkspace() ? state.runName : null;
  $("#stages-target").textContent = workspace ? `on ${workspace}` : "select or create a workspace";
  $("#run-all").disabled = !workspace;
  if (!state.stages) { box.innerHTML = `<div class="empty">${state.stagesError ? "stages unavailable: " + escapeHtml(state.stagesError) : "loading stage schemas…"}</div>`; delete box.dataset.stages; return; }
  const key = state.stages.map((s) => s.name).join(",");
  if (box.dataset.stages !== key) buildStages(box, key);
  for (const stage of state.stages) {
    const run = box.querySelector(`.stage[data-stage="${stage.name}"] .stage-run`);
    if (run) run.disabled = !workspace;
    renderStageMeta(stage.name);
  }
  if (typeof workflowRun !== "undefined") workflowRun.render();
}

function rebuildStages() {
  delete $("#stages").dataset.stages;
  renderStages();
}

function buildStages(box, key) {
  const workspace = isWorkspace() ? state.runName : null;
  box.innerHTML = "";
  box.dataset.stages = key;
  const expanded = readStorage("poseViewer.stageOpen", {});
  const included = readStorage("poseViewer.stageInclude", {});
  for (const stage of state.stages.filter((s) => s.name !== "export")) {
    const node = document.createElement("div");
    node.className = "stage";
    node.dataset.stage = stage.name;
    const head = document.createElement("div");
    head.className = "stage-head";
    const checked = included[stage.name] ?? stage.name !== "fixed_body";
    const label = stage.name === "fixed_body" ? "Fixed body (optional)" : stage.name;
    head.innerHTML = `<input type="checkbox" class="stage-include" title="include in pipeline" aria-label="Include ${escapeHtml(label)} in pipeline" ${checked ? "checked" : ""}><button type="button" class="stage-toggle${expanded[stage.name] ? " active" : ""}" aria-expanded="${!!expanded[stage.name]}" title="parameters">${expanded[stage.name] ? "▾" : "▸"}</button><b>${escapeHtml(label)}</b><span class="stage-meta"></span><button type="button" class="stage-run primary" ${workspace ? "" : "disabled"}>Run</button>`;
    node.appendChild(head);
    if (stage.name === "fixed_body") {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = "Run after reviewing and repairing poses. Freezes length and widths from clear frames across this workspace, and adds a separate Fixed body layer. Rebuild after edits.";
      node.appendChild(note);
    }
    const form = document.createElement("div");
    form.className = "stage-form";
    form.hidden = !expanded[stage.name];
    if (!stage.params.length) form.innerHTML = '<div class="empty">no parameters</div>';
    for (const param of stage.params) form.appendChild(renderParamField(stage.name, param));
    if (stage.params.length) {
      const reset = document.createElement("button");
      reset.type = "button"; reset.className = "stage-reset"; reset.textContent = "defaults";
      reset.addEventListener("click", () => { delete state.stageValues[stage.name]; writeStorage("poseViewer.stageParams", state.stageValues); rebuildStages(); });
      form.appendChild(reset);
    }
    node.appendChild(form);
    head.querySelector(".stage-toggle").addEventListener("click", () => { form.hidden = !form.hidden; head.querySelector(".stage-toggle").textContent = form.hidden ? "▸" : "▾"; expanded[stage.name] = !form.hidden; head.querySelector(".stage-toggle").classList.toggle("active", !form.hidden); head.querySelector(".stage-toggle").setAttribute("aria-expanded", String(!form.hidden)); writeStorage("poseViewer.stageOpen", expanded); });
    head.querySelector(".stage-include").addEventListener("change", (e) => { included[stage.name] = e.target.checked; writeStorage("poseViewer.stageInclude", included); });
    head.querySelector(".stage-run").addEventListener("click", () => runStage(stage.name));
    box.appendChild(node);
  }
}

function selectedJobGpu() {
  const value = readStorage("poseApp.jobGpu", null);
  return Number.isInteger(value) && (state.info?.gpus || []).includes(value) ? value : null;
}

function renderJobGpuControls() {
  const gpus = state.info?.gpus || [], selected = selectedJobGpu();
  for (const id of ["job-gpu", "jobs-gpu"]) {
    const select = $("#" + id);
    if (!select) continue;
    select.replaceChildren(new Option(gpus.length ? "Automatic" : "CPU", ""));
    for (const gpu of gpus) select.add(new Option(`GPU ${gpu}`, String(gpu)));
    select.value = selected === null ? "" : String(selected);
    select.disabled = !gpus.length;
    select.onchange = () => {
      writeStorage("poseApp.jobGpu", select.value === "" ? null : Number(select.value));
      renderJobGpuControls();
    };
  }
}

async function submitStage(workspace, stage, gpu = selectedJobGpu()) {
  const params = collectParams(stage);
  const record = await post("/api/jobs", { kind: "stage", workspace, stage, params, gpu });
  await loadJobs();
  return record;
}

async function runStage(stage) {
  if (!isWorkspace()) { setStatus("select a workspace first", "error"); return; }
  const workspace = state.runName;
  if (typeof maskEditor !== "undefined" && !await maskEditor.ensureSavedForRerun()) return;
  if (workspace !== state.runName || !isWorkspace()) return;
  try {
    const record = await submitStage(workspace, stage);
    setStatus(`queued ${stage} on ${state.runName} as ${record.id}`, "ok");
    showTab("pipeline");
    openRightTab("jobs", { refresh: false });
  } catch (error) { setStatus(error.message, "error"); }
}

// "Run all" queues the included stages one after another: the next stage is
// submitted when the previous job reports done, so stages that depend on
// each other never run at once on different GPUs.  The chain lives in this
// tab's sessionStorage: a reload of the tab continues it, but a second tab
// on the same server never drives it too (two drivers would submit every
// stage twice).  Before submitting, a stage already queued or running on the
// workspace is adopted instead of duplicated.
function readSession(key, fallback) {
  try { const raw = sessionStorage.getItem(key); return raw === null ? fallback : JSON.parse(raw); } catch (error) { return fallback; }
}

function writeSession(key, value) {
  try { sessionStorage.setItem(key, JSON.stringify(value)); } catch (error) { /* blocked storage */ }
}

function saveChain() { writeSession("poseViewer.chain", state.chain); }

async function runAll() {
  if (!isWorkspace()) { setStatus("select a workspace first", "error"); return; }
  const workspace = state.runName;
  if (typeof maskEditor !== "undefined" && !await maskEditor.ensureSavedForRerun()) return;
  if (workspace !== state.runName || !isWorkspace()) return;
  if (state.chain) { setStatus("A pipeline is already running; wait for it to finish or stop it in Jobs", "error"); return; }
  const queue = [...document.querySelectorAll("#stages .stage")].filter((n) => n.dataset.stage !== "export" && n.querySelector(".stage-include").checked).map((n) => n.dataset.stage);
  if (!queue.length) { setStatus("no stages included", "error"); return; }
  state.chain = { workspace: state.runName, queue, current: null, stages: [...queue], gpu: selectedJobGpu() };
  if (typeof workflowRun !== "undefined") workflowRun.started(state.runName, queue);
  openRightTab("jobs", { refresh: false });
  saveChain();
  advanceChain();
}

function cancelChain(reason) {
  if (!state.chain) return;
  const remaining = state.chain.queue.length;
  state.chain = null;
  saveChain();
  renderChain();
  if (reason) setStatus(`${reason}; ${remaining} stage${remaining === 1 ? "" : "s"} not run`, "error");
}

function activeJobOf(stage, workspace) {
  return state.jobs.find((j) => jobActive(j) && (j.spec || {}).workspace === workspace && jobStage(j) === stage) || null;
}

async function advanceChain() {
  const chain = state.chain;
  if (!chain || chain.current) return;
  if (!chain.queue.length) { if (typeof workflowRun !== "undefined") workflowRun.completed(chain.workspace); state.chain = null; saveChain(); renderChain(); setStatus(`all stages finished on ${chain.workspace}`, "ok"); return; }
  const stage = chain.queue.shift();
  const existing = activeJobOf(stage, chain.workspace);
  if (existing) {
    chain.current = existing.id;
    saveChain();
    setStatus(`run all: ${stage} is already ${existing.state} as ${existing.id}; following it`, "ok");
    renderChain();
    return;
  }
  try {
    const record = await submitStage(chain.workspace, stage, chain.gpu ?? null);
    chain.current = record.id;
    saveChain();
    setStatus(`run all: ${stage} queued as ${record.id}, ${chain.queue.length} to follow`, "ok");
  } catch (error) {
    cancelChain(`run all: could not submit ${stage} (${error.message})`);
  }
  renderChain();
}

function renderChain() {
  const node = $("#chain");
  const chain = state.chain;
  if (!chain) { node.hidden = true; node.innerHTML = ""; return; }
  node.hidden = false;
  const current = chain.current ? state.jobs.find((j) => j.id === chain.current) : null;
  node.innerHTML = `<span>run all on <b>${escapeHtml(chain.workspace)}</b>: ${current ? `${escapeHtml(jobStage(current))} ${current.state}` : "submitting"} → ${chain.queue.map(escapeHtml).join(" → ") || "end"}</span><button type="button" id="chain-stop">stop</button>`;
  $("#chain-stop").addEventListener("click", () => cancelChain("run all stopped"));
}

// Follow the chain's current job: done -> next stage, failed or cancelled -> stop.
function tickChain() {
  const chain = state.chain;
  if (!chain || !chain.current) return;
  const job = state.jobs.find((j) => j.id === chain.current);
  if (!job) return;
  if (job.state === "done") { chain.current = null; advanceChain(); }
  else if (job.state === "failed" || job.state === "cancelled") cancelChain(`run all: ${jobStage(job)} ${job.state}`);
  else renderChain();
}

// ---------------------------------------------------------------- jobs

const JOBS_POLL_ACTIVE_MS = 2000;
const JOBS_POLL_IDLE_MS = 15000;
let jobsTimer = null;

const jobActive = (job) => job.state === "queued" || job.state === "running";

function renderJobsBadge() {
  const running = state.jobs.length ? state.jobs.filter(jobActive).length : (state.info && state.info.jobs_running) || 0;
  $("#jobs-badge").textContent = running ? String(running) : "";
  $("#jobs-badge").hidden = !running;
}

async function loadJobs() {
  let jobs;
  try {
    const payload = await api("/api/jobs");
    jobs = Array.isArray(payload) ? payload : payload.jobs || [];
  } catch (error) {
    $("#jobs").innerHTML = `<div class="empty">jobs unavailable: ${escapeHtml(error.message)}</div>`;
    return;
  }
  const finished = [];
  for (const job of jobs) {
    const before = state.jobStates.get(job.id);
    if (before !== undefined && before !== job.state && !jobActive(job)) finished.push(job);
    state.jobStates.set(job.id, job.state);
  }
  if (typeof corpusUI !== "undefined") corpusUI.onJobs(finished);
  state.jobs = jobs;
  renderJobs();
  window.dispatchEvent(new CustomEvent("workflow:jobs"));
  renderJobsBadge();
  if (state.stages) for (const stage of state.stages) renderStageMeta(stage.name);
  if (typeof renderCandidateJobs === "function") renderCandidateJobs();
  tickChain();
  // The interval follows what is running now, so a job submitted while idle is polled at once.
  scheduleJobsPoll();
  await refreshLogs();
  // A job that just finished on the current workspace: show its results; one that failed or was cancelled: say so.
  const mine = finished.filter((j) => (j.spec || {}).workspace && isWorkspace() && j.spec.workspace === state.runName);
  const broken = mine.filter((j) => j.state === "failed" || j.state === "cancelled");
  if (broken.length) {
    setStatus(broken.map((j) => `${jobStage(j)} ${j.id} ${j.state}${j.error || j.message ? `: ${j.error || j.message}` : ""}`).join(" · "), "error");
  }
  if (mine.some((j) => j.state === "done")) {
    if (!broken.length) setStatus(`${mine.filter((j) => j.state === "done").map((j) => jobStage(j)).join(", ")} finished on ${state.runName}; refreshing`, "ok");
    await reloadSource();
  }
  if (finished.length) refreshCatalog().catch(() => {});
}

// Polls fast while anything runs, slowly when idle, and not at all while the
// tab is hidden (a return to the tab refreshes at once).
function scheduleJobsPoll() {
  clearTimeout(jobsTimer);
  if (document.hidden) return;
  const active = state.jobs.some(jobActive) || (state.chain && !state.chain.current);
  jobsTimer = setTimeout(() => { loadJobs().catch(() => {}).then(scheduleJobsPoll); }, active ? JOBS_POLL_ACTIVE_MS : JOBS_POLL_IDLE_MS);
}

function jobElapsed(job) {
  if (!job.started_at) return null;
  const start = new Date(job.started_at).getTime();
  const end = job.finished_at ? new Date(job.finished_at).getTime() : Date.now();
  return (end - start) / 1000;
}

// Job ids whose result JSON is shown in place of the log.
const openResults = new Set();

// The list is updated in place, keyed by job id: nodes are created and
// removed only when the id set changes, so an expanded log or result stays
// put across the poll and finished logs are fetched once.
function renderJobs() {
  const box = $("#jobs");
  const mine = $("#jobs-mine").checked && isWorkspace();
  const filter = $("#jobs-state").value;
  const jobs = state.jobs.filter((j) => (!filter || j.state === filter) && (!mine || (j.spec || {}).workspace === state.runName)).slice(0, 60);
  $("#jobs-summary").textContent = `${state.jobs.filter(jobActive).length} active · ${state.jobs.length} total`;
  if (!jobs.length) { box.innerHTML = `<div class="empty">${mine ? "no jobs on this workspace" : "no jobs"}</div>`; renderChain(); return; }
  for (const empty of box.querySelectorAll(".empty")) empty.remove();
  const wanted = new Set(jobs.map((j) => j.id));
  for (const node of [...box.querySelectorAll(".job")]) if (!wanted.has(node.dataset.job)) node.remove();
  let previous = null;
  for (const job of jobs) {
    let node = box.querySelector(`.job[data-job="${job.id}"]`);
    if (!node) node = createJobNode(job.id);
    updateJobNode(node, job);
    const anchor = previous ? previous.nextSibling : box.firstChild;
    if (node !== anchor) box.insertBefore(node, anchor);
    previous = node;
  }
  renderChain();
}

function createJobNode(id) {
  const node = document.createElement("div");
  node.className = "job";
  node.dataset.job = id;
  node.innerHTML = '<div class="job-head"><b></b><span class="job-ws"></span><span class="job-state"></span></div>' +
    '<div class="bar"><div></div></div>' +
    '<div class="job-line"><span data-ident></span><span data-elapsed></span></div>' +
    '<div class="job-msg"></div>' +
    '<div class="row"><button type="button" data-log>log</button><button type="button" data-cancel hidden>Cancel</button><button type="button" data-retry hidden>Retry</button><button type="button" data-result hidden>result</button></div>' +
    '<pre class="job-log" hidden></pre>';
  node.querySelector("[data-log]").addEventListener("click", () => toggleLog(id));
  node.querySelector("[data-cancel]").addEventListener("click", () => cancelJob(id));
  node.querySelector("[data-retry]").addEventListener("click", () => retryJob(id));
  node.querySelector("[data-result]").addEventListener("click", () => toggleResult(id));
  return node;
}

function updateJobNode(node, job) {
  const spec = job.spec || {};
  const frames = spec.frames ? ` · frames ${spec.frames[0]}–${spec.frames[1]}` : "";
  const elapsed = jobElapsed(job);
  const progress = Math.max(0, Math.min(1, job.progress || 0));
  node.className = `job ${job.state}`;
  node.querySelector(".job-head b").textContent = jobStage(job);
  const ws = node.querySelector(".job-ws");
  ws.textContent = spec.workspace || ""; ws.title = spec.workspace || "";
  node.querySelector(".job-state").textContent = job.state;
  node.querySelector(".bar div").style.width = `${(100 * progress).toFixed(1)}%`;
  const gpu = job.gpu ?? spec.gpu;
  node.querySelector("[data-ident]").textContent = `${job.id}${gpu !== null && gpu !== undefined ? ` · GPU ${gpu}` : ""}${frames}`;
  node.querySelector("[data-elapsed]").textContent = `${jobActive(job) ? `${Math.round(100 * progress)}%` : ""}${elapsed !== null ? " · " + fmtDuration(elapsed) : ""}`;
  node.querySelector(".job-msg").textContent = job.error || job.message || (job.state === "queued" ? `queued ${fmtTime(job.created_at)}` : "");
  const showLog = state.openLogs.has(job.id), showResult = openResults.has(job.id);
  node.querySelector("[data-log]").textContent = showLog ? "hide log" : "log";
  for (const [selector, active] of [["[data-log]", showLog], ["[data-result]", showResult]]) {
    const button = node.querySelector(selector);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  node.querySelector("[data-cancel]").hidden = !jobActive(job);
  node.querySelector("[data-retry]").hidden = jobActive(job);
  node.querySelector("[data-retry]").textContent = job.state === "done" ? "Run again" : "Retry";
  const result = node.querySelector("[data-result]");
  result.hidden = !(job.result && job.state === "done");
  result.textContent = showResult ? "hide result" : "result";
  const pre = node.querySelector(".job-log");
  pre.hidden = !(showLog || showResult);
  if (showResult) pre.textContent = JSON.stringify(job.result, null, 2);
  else if (!showLog) pre.textContent = "";
}

async function retryJob(id) {
  const job = state.jobs.find(j => j.id === id);
  if (!job || jobActive(job)) return;
  const button = document.querySelector(`[data-job="${id}"] [data-retry]`);
  if (button) button.disabled = true;
  try {
    const record = await post(`/api/jobs/${encodeURIComponent(id)}/retry`, {gpu: job.spec.gpus === 0 ? null : selectedJobGpu()});
    await loadJobs();
    setStatus(`queued retry ${record.id}${record.spec.gpu === null ? "" : ` on GPU ${record.spec.gpu}`}`, "ok");
  } catch (error) { setStatus(error.message, "error"); }
  finally { if (button) button.disabled = false; }
}

async function toggleLog(id) {
  if (state.openLogs.has(id)) state.openLogs.delete(id); else { state.openLogs.add(id); openResults.delete(id); }
  renderJobs();
  await refreshLogs(true);
}

function toggleResult(id) {
  if (openResults.has(id)) openResults.delete(id); else { openResults.add(id); state.openLogs.delete(id); }
  renderJobs();
}

// Logs of the expanded jobs; running ones refresh with every poll.
async function refreshLogs(all = false) {
  for (const id of state.openLogs) {
    const job = state.jobs.find((j) => j.id === id);
    const pre = document.querySelector(`.job[data-job="${id}"] .job-log`);
    if (!pre || (!all && job && !jobActive(job) && pre.textContent)) continue;
    try {
      const payload = await api(`/api/jobs/${encodeURIComponent(id)}/log?tail=200`);
      const text = typeof payload === "string" ? payload : payload.log !== undefined ? payload.log : payload.text || JSON.stringify(payload);
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
      pre.textContent = text || "(empty log)";
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    } catch (error) { pre.textContent = `log unavailable: ${error.message}`; }
  }
}

async function cancelJob(id) {
  try {
    await post(`/api/jobs/${encodeURIComponent(id)}/cancel`, {});
    setStatus(`cancelled ${id}`, "ok");
    await loadJobs();
  } catch (error) { setStatus(error.message, "error"); }
}

// Called by selectSource once a source is loaded.
function onSourceChanged() {
  if (typeof corpusUI !== "undefined") corpusUI.sourceChanged();
  renderStages();
  renderJobs();
  if (state.recording) renderRecordingDetail();
  if (typeof regionsOnSourceChanged === "function") regionsOnSourceChanged();
  window.dispatchEvent(new CustomEvent("workflow:source"));
}

function initPanels() {
  initTabs();
  state.chain = readSession("poseViewer.chain", null);
  if (state.chain) { state.chain.queue = state.chain.queue.filter((stage) => stage !== "export"); saveChain(); }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearTimeout(jobsTimer);
    else loadJobs().catch(() => {}).then(scheduleJobsPoll);
  });
  $("#recording-filter").addEventListener("input", renderRecordings);
  $("#recordings-rescan").addEventListener("click", () => loadRecordings(true));
  $("#explorer-toggle").addEventListener("click", () => toggleExplorer());
  $("#explorer-go").addEventListener("click", () => browseTo($("#explorer-path").value.trim() || null));
  $("#explorer-path").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); browseTo($("#explorer-path").value.trim() || null); } });
  $("#explorer-up").addEventListener("click", () => { const parent = $("#explorer-up").dataset.parent; if (parent) browseTo(parent); });
  $("#explorer-register").addEventListener("click", registerRecording);
  $("#explorer-all").addEventListener("change", () => browseTo(explorer.path));
  $("#thumb-frame").addEventListener("change", () => { if (state.recording) setThumbnail(state.recording, thumbFrame()); });
  $("#ws-first").addEventListener("input", syncWorkspaceName);
  $("#ws-last").addEventListener("input", syncWorkspaceName);
  $("#ws-step").addEventListener("input", syncWorkspaceName);
  $("#ws-name").addEventListener("input", () => { $("#ws-name").dataset.auto = $("#ws-name").value.trim() ? "0" : "1"; syncWorkspaceName(); });
  $("#ws-create").addEventListener("click", createWorkspace);
  $("#import-run").addEventListener("click", importSelectedRun);
  $("#run-all").addEventListener("click", runAll);
  $("#jobs-state").addEventListener("change", () => loadJobs());
  $("#jobs-mine").addEventListener("change", renderJobs);
  $("#jobs-refresh").addEventListener("click", () => loadJobs());
  loadStages();
  if (typeof initRegions === "function") initRegions();
  loadJobs().catch(() => {}).then(scheduleJobsPoll);
}

// Selection in Open is reviewable before resuming or making a separate copy.
function renderOpenSelection() {
  const button = $("#open-selected");
  if (!button) return;
  const selected = parseSourceKey($("#run").value);
  const entry = selected && (selected.kind === "workspace" ? workspaceEntry(selected.name) : runEntry(selected.name));
  button.disabled = !entry || !!entry.error;
  button.textContent = selected?.kind === "run" ? "Open run read-only" : "Resume workspace";
  button.classList.toggle("primary", selected?.kind !== "run");
  $("#import-run").classList.toggle("primary", selected?.kind === "run");
  $("#import-run").hidden = !entry || selected.kind !== "run";
  $("#run-info").textContent = entry ? `${selected.name} · ${sourceOptionTitle(selected.kind, entry)}${selected.kind === "run" ? "\nCreate editable copy preserves the original run." : "\nResume restores saved workspace context."}` : "Select a workspace or run above.";
  button.onclick = async () => {
    const source = parseSourceKey($("#run").value);
    if (source && await selectSource(source.kind, source.name) === true) showTab(state.activeTask || "inspect");
  };
}
function importSelectedRun() {
  const selected = parseSourceKey($("#run").value);
  if (selected?.kind === "run") return importRun(selected.name);
}
