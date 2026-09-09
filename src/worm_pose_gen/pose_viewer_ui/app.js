"use strict";

// Entry point of the pose viewer: wires the controls to the functions the
// other scripts define (api.js, layers.js, charts.js, viewer.js, panels.js,
// loaded before this file) and boots the page from /api/state.

function bindEvents() {
  $("#run").addEventListener("change", (e) => { const source = parseSourceKey(e.target.value); if (source) selectSource(source.kind, source.name); });
  $("#run-filter").addEventListener("input", renderRunList);
  $("#rescan").addEventListener("click", async () => {
    try {
      const info = await refreshCatalog(true);
      const added = info.added === undefined ? "" : `${info.added} new run${info.added === 1 ? "" : "s"} found · `;
      setStatus(`${added}${state.workspaces.length} workspaces, ${state.runs.length} runs`, "ok");
    } catch (error) { setStatus(error.message, "error"); }
  });
  $("#compare").addEventListener("change", (e) => selectCompare(e.target.value));
  $("#go").addEventListener("click", () => { if (!state.run) return; const r = state.run.series.frame_index.indexOf(parseInt($("#frame-index").value, 10)); if (r >= 0) showRow(r, { keepView: true }); else setStatus("frame not in this source", "error"); });
  $("#frame-index").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#go").click(); });
  $("#prev").addEventListener("click", () => step(-1));
  $("#next").addEventListener("click", () => step(1));
  $("#play").addEventListener("click", togglePlay);
  $("#worst").addEventListener("change", (e) => { if (e.target.value !== "") showRow(parseInt(e.target.value, 10), { keepView: true }); });
  for (const button of document.querySelectorAll("[data-jump]")) {
    const [kind, direction] = button.dataset.jump.split(":");
    button.addEventListener("click", () => jump(kind, parseInt(direction, 10)));
  }
  $("#toggle-raw").addEventListener("click", () => { state.showRaw = !state.showRaw; $("#toggle-raw").classList.toggle("active", state.showRaw); showRow(state.row, { keepView: true, keepStarts: true, immediate: true }); });
  $("#fit-view").addEventListener("click", fitView);
  $("#starts").addEventListener("click", computeStarts);
  $("#threshold").addEventListener("input", (e) => { $("#threshold-value").textContent = e.target.value; if ($("#threshold-on").checked) showRow(state.row, { keepView: true, keepStarts: true }); });
  $("#threshold-on").addEventListener("change", (e) => { $("#threshold-value").textContent = e.target.checked ? $("#threshold").value : "run"; showRow(state.row, { keepView: true, keepStarts: true, immediate: true }); });
  $("#note-save").addEventListener("click", saveNote);
  $("#note-comment").addEventListener("keydown", (e) => { if (e.key === "Enter") saveNote(); e.stopPropagation(); });
  $("#extra-series").addEventListener("change", (e) => { state.extra = e.target.value; drawCharts(); });
  $("#show-independent").addEventListener("change", drawCharts);
  $("#show-compare").addEventListener("change", drawCharts);
  $("#jump-iou").addEventListener("change", drawCharts);
  $("#timeline-toggle").addEventListener("click", () => togglePanel("bottom"));
  // Phase 2 edits.
  $("#flip-frame").addEventListener("click", flipFrame);
  $("#flip-segment").addEventListener("click", flipSegment);
  $("#edit-undo").addEventListener("click", () => undoEdit());
  $("#edits-refresh").addEventListener("click", loadEdits);

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const before = toImage(event.clientX, event.clientY);
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    state.view.scale = Math.max(0.05, Math.min(40, state.view.scale * factor));
    const rect = canvas.getBoundingClientRect();
    state.view.tx = event.clientX - rect.left - before.x * state.view.scale;
    state.view.ty = event.clientY - rect.top - before.y * state.view.scale;
    state.userView = true;
    draw();
  }, { passive: false });
  canvas.addEventListener("pointerdown", (event) => { state.panning = true; state.last = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); canvas.style.cursor = "grabbing"; });
  canvas.addEventListener("pointermove", (event) => {
    if (state.panning && state.last) {
      state.view.tx += event.clientX - state.last.x; state.view.ty += event.clientY - state.last.y;
      state.last = { x: event.clientX, y: event.clientY }; state.userView = true; draw();
    } else if (state.decoded) {
      const p = toImage(event.clientX, event.clientY);
      const x = Math.floor(p.x), y = Math.floor(p.y);
      const d = state.decoded;
      if (x >= 0 && y >= 0 && x < d.width && y < d.height) {
        const i = y * d.width + x;
        const bits = [`x ${x} y ${y}`];
        if (d.probability) bits.push(`p ${(d.probability[i] / 255).toFixed(2)}`);
        if (d.mask_final) bits.push(d.mask_final[i] ? "mask" : "");
        if (d.tube) bits.push(d.tube[i] ? "tube" : "");
        canvas.title = bits.filter(Boolean).join(" · ");
      }
    }
  });
  canvas.addEventListener("pointerup", () => { state.panning = false; state.last = null; canvas.style.cursor = "grab"; });
  // Panels resize the stage and the charts without a window resize.
  let pending = null;
  const relayout = () => {
    if (pending) return;
    pending = requestAnimationFrame(() => { pending = null; resizeCanvas(); drawCharts(); if (state.frame) { drawWidthChart(); drawCurvatureChart(); } });
  };
  new ResizeObserver(relayout).observe($("#stage"));
  new ResizeObserver(relayout).observe($("#charts"));
  new ResizeObserver(relayout).observe($("#details"));
  window.addEventListener("resize", relayout);
  initSplitters();
  initPanels();

  window.addEventListener("keydown", (event) => {
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") { if (event.key === "Escape") document.activeElement.blur(); return; }
    if ((event.ctrlKey || event.metaKey) && !event.shiftKey && !event.altKey && (event.key === "z" || event.key === "Z")) {
      event.preventDefault();
      if (editsSupported()) undoEdit(); else setStatus("undo needs a workspace on the app server", "error");
      return;
    }
    const big = event.ctrlKey ? 100 : event.shiftKey ? 10 : 1;
    switch (event.key) {
      case "ArrowLeft": step(-big); event.preventDefault(); break;
      case "ArrowRight": step(big); event.preventDefault(); break;
      case " ": togglePlay(); event.preventDefault(); break;
      case "[": jump("flag", -1); break;
      case "]": jump("flag", 1); break;
      case ",": jump("iou", -1); break;
      case ".": jump("iou", 1); break;
      case "f": case "F": $("#toggle-raw").click(); break;
      case "0": fitView(); break;
      case "n": case "N": showTab("view"); $("#note-comment").focus(); event.preventDefault(); break;
      case "s": case "S": computeStarts(); break;
      default:
        if (/^[1-9]$/.test(event.key)) toggleLayer(parseInt(event.key, 10) - 1);
    }
  });
}

// The source to open first: the URL hash, else the newest workspace, else the newest run.
function initialSource(wanted) {
  if (wanted.ws && workspaceEntry(wanted.ws)) return { kind: "workspace", name: wanted.ws };
  if (wanted.run && runEntry(wanted.run)) return { kind: "run", name: wanted.run };
  if (state.workspaces.length) return { kind: "workspace", name: state.workspaces[0].name };
  if (state.runs.length) return { kind: "run", name: state.runs[0].name };
  return null;
}

async function boot() {
  renderLayers();
  renderNoteTags();
  bindEvents();
  resizeCanvas();
  try {
    await refreshCatalog(false);
    await loadNotes();
    const errors = Object.entries(state.info.errors || {});
    if (errors.length) setStatus(`${errors.length} run directories skipped: ${errors.map(([p, e]) => `${p.split("/").pop()} (${e})`).join("; ")}`, "error");
    const wanted = parseHash();
    const initial = initialSource(wanted);
    if (initial) await selectSource(initial.kind, initial.name, wanted.frame);
    else { setStatus("no workspaces or runs yet: pick a recording in the Recordings tab and create a workspace", "error"); showTab("data"); }
  } catch (error) {
    setStatus(error.message, "error");
  }
}

boot();
