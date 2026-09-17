"use strict";

// The import handoff and whole-workspace pipeline share the existing stage
// forms and job queue. This controller never changes stage parameter defaults.
const workflowRun = (() => {
  let mounted = false;
  const runs = new Map();
  let validationKey = null, validation = null, validationRequest = 0;
  const selectedCheckpoint = () => (state.stageValues.segment || {}).checkpoint || state.run?.selected_checkpoint || state.run?.summary?.selected_checkpoint || state.run?.workspace_info?.settings?.checkpoint || state.run?.entry?.checkpoint_path || state.info?.fallback_checkpoint || '';
  async function checkCheckpoint(force = false) {
    if (!serverIsApp() || !isWorkspace() || !state.run) return;
    const checkpoint = selectedCheckpoint(), key = `${currentSourceKey()}:${checkpoint}`;
    if (!force && validationKey === key) return;
    validationKey = key; validation = null;
    const token = ++validationRequest;
    render();
    try {
      const [availability, catalog] = await Promise.all([
        api(`/api/checkpoints/availability?checkpoint=${encodeURIComponent(checkpoint)}`), api('/api/checkpoints')
      ]);
      if (token !== validationRequest || key !== `${currentSourceKey()}:${selectedCheckpoint()}`) return;
      validation = availability;
      const select = $('#workflow-model-select'); select.replaceChildren();
      for (const model of catalog.checkpoints || []) {
        if (model.exists === false) continue;
        const option = document.createElement('option'); option.value = model.path;
        option.textContent = model.label || model.path; select.append(option);
      }
      if ([...select.options].some(o => o.value === checkpoint)) select.value = checkpoint;
    } catch (error) {
      if (token !== validationRequest) return;
      validation = {available:false, reason:`Could not check model availability: ${error.message}`};
    }
    render();
  }
  async function chooseCheckpoint() {
    if (!maskEditor.beforeMutation()) return;
    const checkpoint = $('#workflow-model-select').value, source = currentSourceKey();
    if (!checkpoint) return;
    $('#workflow-model-use').disabled = true;
    try {
      await post(`/api/workspaces/${encodeURIComponent(state.runName)}/checkpoint`, {checkpoint});
      if (source !== currentSourceKey()) return;
      for (const stage of state.stages || []) if ((stage.params || []).some(p => p.name === 'checkpoint')) {
        state.stageValues[stage.name] = {...state.stageValues[stage.name], checkpoint};
      }
      writeStorage('poseViewer.stageParams', state.stageValues);
      state.frameCache.clear(); maskEditor.invalidate();
      await reloadSource(); rebuildStages(); await checkCheckpoint(true);
    } catch (error) { $('#workflow-run-reason').textContent = error.message; }
    finally { $('#workflow-model-use').disabled = !$('#workflow-model-select').options.length; }
  }
  function resetImport() {
    if (!mounted) return;
    $("#ws-range-mode").value = "entire";
    renderImport();
  }
  function renderImport() {
    if (!mounted || !state.recording) return;
    const rec = state.recording, entire = $("#ws-range-mode").value === "entire";
    const known = Number.isInteger(rec.frames) && rec.frames > 0;
    if (entire) {
      $("#ws-first").value = 0;
      $("#ws-last").value = known ? rec.frames - 1 : "";
      $("#ws-step").value = 1;
    }
    for (const id of ["ws-first", "ws-last", "ws-step"]) $("#" + id).disabled = entire;
    const first = Number($("#ws-first").value), last = Number($("#ws-last").value), step = Number($("#ws-step").value);
    const valid = known && ["ws-first", "ws-last", "ws-step"].every(id => $("#" + id).value.trim() !== "") && Number.isInteger(first) && Number.isInteger(last) && Number.isInteger(step) && first >= 0 && last >= first && last < rec.frames && step >= 1;
    const count = valid ? Math.floor((last - first) / step) + 1 : null;
    const fps = Number(rec.fps || rec.frame_rate);
    $("#ws-range-summary").textContent = valid ? `Frames ${first.toLocaleString()}–${last.toLocaleString()} (inclusive) · ${count.toLocaleString()} frames${step > 1 ? ` · every ${step} frames` : ""}${fps > 0 ? ` · ${fmtDuration((last - first + 1) / fps)}` : ""}` : known ? "Choose whole-number bounds inside this recording and a step of at least 1." : "Recording frame count unavailable.";
    $("#ws-create").disabled = !valid || !rec.readable || !serverIsApp();
  }
  function started(workspace, stages) {
    runs.set(workspace, { stages: [...stages], done: false });
    render();
  }
  function completed(workspace) {
    const run = runs.get(workspace) || {};
    run.done = true;
    runs.set(workspace, run);
    // advanceChain clears the chain immediately after this callback.
    queueMicrotask(render);
  }
  function render() {
    if (!mounted) return;
    const workspace = isWorkspace() && state.run;
    const node = $("#workflow-run-overview");
    node.hidden = !workspace;
    if (!workspace) return;
    const entry = state.run.entry || {}, summary = state.run.summary || {};
    const series = state.run.series || {}, frames = series.frame_index || [];
    const mine = state.jobs.filter(j => (j.spec || {}).workspace === state.runName && j.spec.kind === "stage" && jobStage(j) !== "export");
    const active = mine.some(jobActive) || (state.chain && state.chain.workspace === state.runName);
    const record = runs.get(state.runName);
    const processed = (series.fitted || []).some(Boolean);
    $("#workflow-run-status").textContent = active ? "Pipeline in progress" : record && record.done ? "Pipeline complete" : processed ? "Configure another pipeline run" : "This recording has not been processed yet";
    const range = frames.length ? `Workspace frames ${frames[0].toLocaleString()}–${frames[frames.length - 1].toLocaleString()} · ${frames.length.toLocaleString()} frames${entry.step > 1 ? ` · step ${entry.step}` : ""}` : "Workspace frame range unavailable";
    const checkpoint = selectedCheckpoint();
    checkCheckpoint();
    const edits = Object.entries(state.stageValues || {}).filter(([stage]) => stage !== "export").reduce((n, [, params]) => n + Object.keys(params || {}).length, 0);
    $("#workflow-run-summary").textContent = `${range}\nCheckpoint: ${checkpoint || "unavailable — configure a segment checkpoint below"}\nConfiguration: ${edits ? `${edits} saved parameter overrides` : "standard stage defaults"}`;
    $("#workflow-inspect-results").hidden = active || !(record && record.done);
    const segmentationIncluded = !!$('#stages [data-stage="segment"] .stage-include:checked');
    const needsCheckpoint = segmentationIncluded && validation?.available !== true;
    $('#workflow-run-reason').textContent = !segmentationIncluded ? '' : !validation ? 'Checking selected model availability…' : !validation.available ? `${validation.reason || 'Model unavailable'} — choose an available checkpoint below, or change the checkpoint in Detailed stage configuration.` : 'Selected model is available. Other stage prerequisites are checked when each job starts.';
    $('#workflow-model-use').disabled = !$('#workflow-model-select').options.length;
    $('#workflow-run-models').hidden = !segmentationIncluded;
    $('#run-all').disabled = !!active || !state.stages || needsCheckpoint;
    const segmentRun = $('#stages [data-stage="segment"] .stage-run');
    if (segmentRun) segmentRun.disabled = !!active || validation?.available !== true;
    const stages = (state.stages || []).filter(s => s.name !== "export");
    $("#workflow-stage-progress").replaceChildren(...stages.map(stage => {
      const item = document.createElement("li"), job = mine.find(j => jobStage(j) === stage.name);
      const queued = state.chain && state.chain.workspace === state.runName && state.chain.queue.includes(stage.name);
      const included = $(`#stages [data-stage="${stage.name}"] .stage-include`);
      const status = queued ? "pending" : job ? job.state : included && !included.checked ? "not included" : stage.name === "segment" && validation?.available !== true ? (validation ? "model unavailable" : "checking model") : "included";
      item.textContent = `${stage.name} · ${status}${status === "running" ? ` ${Math.round((job.progress || 0) * 100)}%` : ""}`;
      item.dataset.state = status;
      return item;
    }));
  }
  function init() {
    if (mounted) return;
    mounted = true;
    const mode = document.createElement("label");
    mode.className = "workflow-range-mode";
    mode.innerHTML = 'Recording range<select id="ws-range-mode"><option value="entire" selected>Entire recording</option><option value="selected">Selected range</option></select>';
    $("#ws-name").closest("label").after(mode);
    const range = document.createElement("p");
    range.id = "ws-range-summary"; range.className = "note"; range.setAttribute("aria-live", "polite");
    $("#ws-name").closest(".form-grid").after(range);
    $("#ws-range-mode").addEventListener("change", () => { renderImport(); syncWorkspaceName(); });
    $("#ws-create").textContent = "Create workspace & configure pipeline";
    const overview = document.createElement("section");
    overview.id = "workflow-run-overview"; overview.className = "group"; overview.hidden = true;
    overview.innerHTML = '<h3 id="workflow-run-status"></h3><p id="workflow-run-summary" class="note"></p><p id="workflow-run-reason" class="note" role="status"></p><div id="workflow-run-models"><label>Available model<select id="workflow-model-select"></select></label><div class="row wrap"><button id="workflow-model-use" type="button">Use selected model</button><button id="workflow-model-refresh" type="button">Check again</button></div></div><ol id="workflow-stage-progress" aria-label="Pipeline stages"></ol><button id="workflow-inspect-results" class="primary" hidden>Inspect results</button>';
    $('#rerun-scope-info').after(overview);
    $('#workflow-model-use').onclick = chooseCheckpoint;
    $('#workflow-model-refresh').onclick = () => checkCheckpoint(true);
    $("#workflow-inspect-results").onclick = () => showTab("inspect");
    $("#run-all").textContent = "Run pipeline";
    $("#run-all").title = "Run the included pipeline stages in order";
    $("#run-all").nextElementSibling.textContent = "Included stages run in order, one job at a time.";
    const advanced = document.createElement("details");
    advanced.id = "workflow-stage-details";
    const heading = document.createElement("summary");
    heading.textContent = "Detailed stage configuration";
    advanced.append(heading);
    $("#stages").before(advanced);
    advanced.append($("#stages"));
    const group = $("#whole-stages .group");
    group.querySelector("h3").firstChild.textContent = "Pipeline configuration ";
    for (const event of ["workflow:source", "workflow:jobs", "workflow:task", "workflow:changed"]) window.addEventListener(event, () => { if (event === "workflow:source" || event === "workflow:changed") validationKey = null; render(); });
    $("#stages").addEventListener("change", () => queueMicrotask(render));
    render();
  }
  return { init, resetImport, renderImport, render, started, completed };
})();
