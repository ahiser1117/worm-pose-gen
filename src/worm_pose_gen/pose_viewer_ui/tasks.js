"use strict";

// Shared shell actions. Feature tabs never own source selection or global keys.
let draftDecisionPromise = null;
function requestDraftDecision() {
  if (draftDecisionPromise) return draftDecisionPromise;
  const dialog = $("#draft-decision");
  draftDecisionPromise = new Promise((resolve) => {
    dialog.returnValue = "stay";
    dialog.addEventListener("close", () => {
      const answer = dialog.returnValue || "stay";
      draftDecisionPromise = null;
      resolve(answer);
    }, { once: true });
    dialog.showModal();
  });
  return draftDecisionPromise;
}

function syncDrawerToggle() {
  for (const button of document.querySelectorAll("[data-drawer]")) {
    const active = state.drawer === button.dataset.drawer && !$("#app-drawer").hidden;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function closeDrawer() {
  $("#app-drawer").hidden = true;
  state.drawer = null;
  syncDrawerToggle();
}

function toggleDrawer(name) {
  if (state.drawer === name && !$("#app-drawer").hidden) closeDrawer();
  else openDrawer(name);
}

function openDrawer(name) {
  if (RIGHT_TABS.includes(name)) { openRightTab(name); return; }
  if (name !== "shortcuts") return;
  $("#app-drawer").hidden = false;
  $("#drawer-title").textContent = "Keyboard shortcuts";
  $("#drawer-shortcuts").hidden = false;
  state.drawer = name;
  syncDrawerToggle();
  if (typeof shortcuts !== "undefined") shortcuts.renderHelp();
}

const RIGHT_TABS = ["statistics", "layers", "jobs", "history"];
let savedWorkspaceInspector = null;
let shellScreen = "workspace";

function openRightTab(name, {open = true, refresh = true} = {}) {
  if (!RIGHT_TABS.includes(name)) name = "statistics";
  state.rightTab = name;
  $("#app").dataset.rightTab = name;
  if (state.screen === "workspace") writeStorage("poseApp.rightTab", name);
  for (const tab of RIGHT_TABS) {
    $(`#right-${tab}`).hidden = tab !== name;
    const button = $(`#right-tab-${tab}`);
    button.classList.toggle("active", tab === name);
    button.setAttribute("aria-selected", String(tab === name));
    button.tabIndex = tab === name ? 0 : -1;
  }
  if (open) { layout.collapsed.right = false; closeDrawer(); }
  applyLayout();
  if (open) saveLayout();
  if (refresh && name === "jobs") loadJobs();
  if (refresh && name === "history") { loadEdits(); loadOutcomes(); }
  requestAnimationFrame(() => {
    resizeCanvas(); draw(); drawCharts();
    if (state.frame && state.rightTab === "statistics") { drawWidthChart(); drawCurvatureChart(); }
  });
}

function taskRenderScope(scope) {
  $("#whole-stages").hidden = scope !== "workspace";
  $("#regions-section").hidden = scope === "workspace";
  $("#regions-section .region-grid").hidden = scope !== "selection";
  $("#region-info").hidden = scope !== "selection";
  $("#region-anchors").hidden = scope !== "selection";
  const current = $("#rerun-current-anchors");
  if (current) current.hidden = scope !== "current";
}

function syncTaskSelection() {
  const series = state.run && state.run.series.frame_index;
  const region = state.region;
  for (const [id, row] of [["selection-first", region && region.first], ["selection-last", region && region.last]]) {
    const node = $(`#${id}`);
    if (node && document.activeElement !== node) node.value = series && row != null ? series[row] : "";
    if (node) node.disabled = !series;
  }
  $("#selection-info").textContent = !series ? "Open a recording or workspace" : region ? `Selected ${series[region.first]}–${series[region.last]} · ${region.last - region.first + 1} sampled frames` : "No range selected";
  $("#workspace-menu").textContent = state.runName ? state.runName : "No workspace open";
  $("#toggle-raw").classList.toggle("active", state.showRaw);
  $("#toggle-raw").setAttribute("aria-pressed", String(state.showRaw));
  $("#active-recording").textContent = state.run && state.run.entry.recording || "";
  if (typeof refreshRerunScope === "function") refreshRerunScope();
  window.dispatchEvent(new CustomEvent("workflow:selection"));
}

function syncWorkspaceContext() {
  const target = typeof maskEditor !== "undefined" && maskEditor.corpusActive() ? maskEditor.getTarget() : null;
  const frame = target ? target.frame ?? target.frame_index : state.run?.series.frame_index[state.row];
  $("#context-frame").textContent = frame == null ? "No frame open" : `${target ? "Corpus label" : "Viewing"} · frame ${frame}`;
  const dirty = typeof maskEditor !== "undefined" && maskEditor.isDirty();
  const save = $("#context-save");
  const text = dirty ? "Unsaved mask draft" : target ? "Independent corpus copy" : isWorkspace() ? "Workspace saved" : state.run ? "Read-only run" : "";
  if (save.textContent !== text) save.textContent = text;
  save.classList.toggle("dirty", !!dirty);
  $("#workspace-context").classList.toggle("corpus-context", !!target);
  if (target) $("#selection-info").textContent = "Workspace selection retained";
  $("#workspace-menu").title = state.runName || "";
  $("#active-recording").title = state.run?.entry.recording || "";
}

function taskShellChanged() {
  if (state.screen !== shellScreen) {
    if (shellScreen === "workspace") savedWorkspaceInspector = {tab: state.rightTab, collapsed: layout.collapsed.right, jobsMine: $("#jobs-mine").checked};
    if (state.screen === "workspace" && savedWorkspaceInspector) {
      layout.collapsed.right = savedWorkspaceInspector.collapsed;
      $("#jobs-mine").checked = savedWorkspaceInspector.jobsMine;
      openRightTab(savedWorkspaceInspector.tab, {open: false, refresh: false});
      saveLayout();
    }
    shellScreen = state.screen;
  }
  dismissToast();
  if ($("#frame-index").hasAttribute("aria-invalid") && state.run) $("#frame-index").value = state.run.series.frame_index[state.row];
  clearFieldError($("#frame-index"), "frame-error");
  syncWorkspaceContext();
}

function rememberViewContext() {
  if (!state.run || (typeof maskEditor !== "undefined" && maskEditor.corpusActive())) return;
  const contexts = readStorage("poseApp.contexts", {});
  contexts[currentSourceKey()] = {
    frame: state.run.series.frame_index[state.row], view: { ...state.view },
    userView: state.userView, timeline: { start: state.timeline.start, end: state.timeline.end },
    region: state.region && { ...state.region }, showRaw: state.showRaw,
    stageValues: state.stageValues, regionParams: state.regionParams, algorithm: state.algorithm,
    layers: LAYERS.map(l => ({ id: l.id, on: l.on, alpha: l.alpha })),
  };
  writeStorage("poseApp.contexts", contexts);
}

function rememberedViewContext(kind, name) {
  return readStorage("poseApp.contexts", {})[sourceKey(kind, name)] || null;
}

function restoreViewContext(context) {
  if (!context || !state.run) return;
  const n = state.run.series.frame_index.length;
  if (context.region && context.region.first >= 0 && context.region.last < n) state.region = { ...context.region };
  if (context.view && Number.isFinite(context.view.scale) && context.view.scale > 0) { state.view = { ...context.view }; state.userView = context.userView; }
  if (context.timeline && context.timeline.start >= 0 && context.timeline.end <= n) Object.assign(state.timeline, context.timeline);
  for (const stored of context.layers || []) { const l = LAYERS.find(item => item.id === stored.id); if (l) { l.on = stored.on; l.alpha = stored.alpha; } }
  state.stageValues = context.stageValues || state.stageValues;
  state.regionParams = context.regionParams || state.regionParams;
  if (context.stageValues) rebuildStages();
  if (context.algorithm && typeof algorithmInfo === "function" && algorithmInfo(context.algorithm)) { state.algorithm = context.algorithm; $("#region-algorithm").value = context.algorithm; }
  if (context.regionParams || context.algorithm) renderAlgorithmForm();
  renderLayers();
  if (typeof renderRegionInfo === "function") renderRegionInfo();
  syncTaskSelection(); draw(); drawCharts();
}

function initTaskShell() {
  for (const button of document.querySelectorAll("[data-screen]")) {
    if (button.tagName === "BUTTON") button.addEventListener("click", () => {
      const screen = button.dataset.screen;
      showTab(["import", "open"].includes(screen) && state.screen === screen ? state.panelReturn || state.activeTask || "inspect" : screen);
    });
  }
  for (const button of document.querySelectorAll("[data-drawer]")) button.addEventListener("click", () => toggleDrawer(button.dataset.drawer));
  for (const button of document.querySelectorAll("[data-close-panel]")) button.onclick = () => showTab(state.panelReturn || state.activeTask || "inspect");
  $("#return-workspace").onclick = async () => {
    if (typeof maskEditor !== "undefined" && maskEditor.corpusActive() && await maskEditor.returnToWorkspace() === false) return;
    showTab(state.activeTask || "inspect");
  };
  $("#toggle-inspector").onclick = () => togglePanel("right");
  $("#review-frame-tools").onclick = () => {
    if (!layout.collapsed.right && state.rightTab === "statistics") togglePanel("right");
    else openRightTab("statistics");
  };
  for (const button of document.querySelectorAll("[data-right-tab]")) {
    button.onclick = () => openRightTab(button.dataset.rightTab);
    button.onkeydown = event => {
      const index = RIGHT_TABS.indexOf(button.dataset.rightTab);
      const next = {ArrowRight: (index + 1) % RIGHT_TABS.length, ArrowLeft: (index + RIGHT_TABS.length - 1) % RIGHT_TABS.length, Home: 0, End: RIGHT_TABS.length - 1}[event.key];
      if (next === undefined) return;
      event.preventDefault();
      openRightTab(RIGHT_TABS[next]);
      $(`#right-tab-${RIGHT_TABS[next]}`).focus();
    };
  }
  openRightTab(readStorage("poseApp.rightTab", "statistics"), {open: false, refresh: false});
  shellScreen = state.screen;
  const selectBounds = () => {
    if (!state.run) return;
    const frames = state.run.series.frame_index;
    const first = frames.indexOf(Number($("#selection-first").value)), last = frames.indexOf(Number($("#selection-last").value));
    if (first < 0 || last < first) { setStatus("Choose first/last frames present in this workspace, in order", "error"); return; }
    setRegionRows(first, last, "explicit selection");
    if (typeof setRerunScope === "function") setRerunScope("selection");
    syncTaskSelection();
  };
  $("#selection-first").onchange = selectBounds;
  $("#selection-last").onchange = selectBounds;
  $("#selection-current").onclick = () => { if (state.run) { setRegionRows(state.row, state.row, "current frame"); syncTaskSelection(); } };
  $("#selection-stretch").onclick = async () => { await proposeRegion(); syncTaskSelection(); };
  window.addEventListener("rerun-scope-changed", event => taskRenderScope(event.detail.scope));
  window.addEventListener("beforeunload", rememberViewContext);
  window.addEventListener("workflow:task", taskShellChanged);
  window.addEventListener("workflow:source", taskShellChanged);
  window.addEventListener("workflow:selection", syncWorkspaceContext);
  window.addEventListener("workflow:mask-state", syncWorkspaceContext);
  const stickyHeight = () => {
    const navigation = $("#tabs").getBoundingClientRect().height;
    const paint = state.screen === "workspace" && state.activeTask === "paint" ? $("#paint-primary").getBoundingClientRect().height : 0;
    $("#sidebar").style.setProperty("--task-sticky-height", `${navigation + paint + 8}px`);
    $("#sidebar").style.setProperty("--task-nav-height", `${navigation}px`);
  };
  const stickyObserver = new ResizeObserver(stickyHeight);
  stickyObserver.observe($("#tabs")); stickyObserver.observe($("#paint-primary"));
  window.addEventListener("workflow:task", stickyHeight);
  $("#screen-tabs").addEventListener("keydown", event => {
    const tabs = [...document.querySelectorAll("#screen-tabs [role=tab]")], index = tabs.indexOf(event.target);
    const next = {ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1}[event.key];
    if (index < 0 || next === undefined) return;
    event.preventDefault(); tabs[next].click(); tabs[next].focus();
  });
  syncTaskSelection();
}
