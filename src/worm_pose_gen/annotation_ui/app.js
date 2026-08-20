"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const canvas = $("#frame-canvas");
const context = canvas.getContext("2d", { alpha: false });
const workArea = $("#work-area");
const emptyState = $("#empty-state");
const loadingFrame = $("#loading-frame");

let sessionState = null;
let currentTask = null;
let currentFrameIndex = null;
let baseImage = null;
let points = [];
let history = [];
let selectedPoint = -1;
let drawing = false;
let dragging = false;
let startedAt = null;
let saving = false;
let frameRequest = 0;

function setMessage(message) {
  $("#save-message").textContent = message || "";
}

function taskKey(task) {
  return `${task.sample_id}:${task.annotation_pass}`;
}

function updateProgress() {
  const complete = sessionState.total_complete;
  const total = sessionState.total_tasks;
  $("#progress-count").textContent = `${complete} / ${total}`;
  $("#progress-label").textContent = `${sessionState.primary_complete}/${sessionState.protocol.primary_count} primary · ${sessionState.repeat_complete}/${sessionState.protocol.repeat_count} repeats`;
  $("#progress-fill").style.width = `${total ? (100 * complete / total) : 0}%`;
}

async function loadState() {
  const response = await fetch("/api/state", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load the annotation session");
  sessionState = await response.json();
  updateProgress();
  const ready = sessionState.tasks.filter((task) => task.status === "ready");
  if (!ready.length) {
    currentTask = null;
    workArea.hidden = true;
    emptyState.hidden = false;
    const finished = sessionState.total_complete === sessionState.total_tasks;
    $("#empty-title").textContent = finished ? "Annotation tranche complete" : "Primary pass complete";
    if (finished) {
      $("#empty-copy").textContent = `All labels are saved to ${sessionState.output_path}.`;
    } else if (sessionState.next_available_at_utc) {
      const when = new Date(sessionState.next_available_at_utc).toLocaleString();
      $("#empty-copy").textContent = `The blind repeat set unlocks ${when}. Close the server and launch the same command later; your progress is already saved.`;
    } else {
      $("#empty-copy").textContent = "No annotation is currently available.";
    }
    return;
  }
  emptyState.hidden = true;
  workArea.hidden = false;
  await loadTask(ready[0]);
}

function resetForm() {
  points = [];
  history = [];
  selectedPoint = -1;
  startedAt = new Date().toISOString();
  $("input[name='trace-state'][value='complete']").checked = true;
  $("input[name='head-tail'][value='ambiguous']").checked = true;
  $("#outside-start").checked = false;
  $("#outside-end").checked = false;
  $("#worm-width").value = "";
  $$("#difficulty-fields input").forEach((input) => { input.checked = false; });
  $("#point-support").value = "supported";
  $("#point-support").disabled = true;
  setMessage("");
  updateTraceControls();
}

async function loadTask(task) {
  currentTask = task;
  currentFrameIndex = task.frame_index;
  resetForm();
  $("#task-title").textContent = `${task.recording} · frame ${task.frame_index}`;
  const passText = task.annotation_pass === "repeat" ? "blind repeat" : "primary";
  $("#task-detail").textContent = `${passText} · ${task.selection_stratum.replaceAll("_", " ")} · ${taskKey(task)}`;
  renderContextStrip();
  await loadFrame(task.frame_index);
}

function renderContextStrip() {
  const strip = $("#context-strip");
  strip.replaceChildren();
  currentTask.temporal_window_indices.forEach((frameIndex) => {
    const button = document.createElement("button");
    const offset = frameIndex - currentTask.frame_index;
    button.type = "button";
    button.textContent = offset === 0 ? `target ${frameIndex}` : `${offset > 0 ? "+" : ""}${offset}`;
    button.className = `${offset === 0 ? "target " : ""}${frameIndex === currentFrameIndex ? "active" : ""}`.trim();
    button.addEventListener("click", () => loadFrame(frameIndex));
    strip.appendChild(button);
  });
}

async function loadFrame(frameIndex) {
  const requestNumber = ++frameRequest;
  currentFrameIndex = frameIndex;
  renderContextStrip();
  loadingFrame.hidden = false;
  const url = `/api/frame?sample_id=${encodeURIComponent(currentTask.sample_id)}&frame_index=${frameIndex}`;
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || "Could not load frame");
  }
  const width = Number(response.headers.get("X-Image-Width"));
  const height = Number(response.headers.get("X-Image-Height"));
  const low = Number(response.headers.get("X-Display-Low"));
  const high = Number(response.headers.get("X-Display-High"));
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (requestNumber !== frameRequest) return;
  const pixels = new Uint8ClampedArray(width * height * 4);
  const scale = Math.max(high - low, 1);
  for (let index = 0; index < bytes.length; index += 1) {
    const value = Math.max(0, Math.min(255, Math.round(255 * (bytes[index] - low) / scale)));
    const offset = 4 * index;
    pixels[offset] = value;
    pixels[offset + 1] = value;
    pixels[offset + 2] = value;
    pixels[offset + 3] = 255;
  }
  canvas.width = width;
  canvas.height = height;
  baseImage = new ImageData(pixels, width, height);
  applyZoom();
  draw();
  loadingFrame.hidden = true;
}

function applyZoom() {
  const zoom = Number($("#zoom").value);
  canvas.style.width = `${100 * zoom}%`;
}

function draw() {
  if (!baseImage) return;
  context.putImageData(baseImage, 0, 0);
  if (currentFrameIndex !== currentTask.frame_index || points.length === 0) return;
  const displayScale = canvas.clientWidth ? canvas.width / canvas.clientWidth : 1;
  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.lineWidth = 3 * displayScale;
  context.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--trace").trim();
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
  context.stroke();

  const radius = 4.5 * displayScale;
  points.forEach((point, index) => {
    context.beginPath();
    context.arc(point.x, point.y, index === selectedPoint ? radius * 1.45 : radius, 0, Math.PI * 2);
    if (index === 0) context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--endpoint-a").trim();
    else if (index === points.length - 1) context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--endpoint-b").trim();
    else context.fillStyle = point.support === "occluded_in_fov" ? "#ff9f43" : "#ffffff";
    context.fill();
    context.lineWidth = displayScale;
    context.strokeStyle = "#071015";
    context.stroke();
  });
  context.restore();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width - 0.001, (event.clientX - rect.left) * canvas.width / rect.width)),
    y: Math.max(0, Math.min(canvas.height - 0.001, (event.clientY - rect.top) * canvas.height / rect.height)),
    support: "supported",
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function nearestPoint(point) {
  let result = -1;
  let best = Infinity;
  points.forEach((candidate, index) => {
    const value = distance(point, candidate);
    if (value < best) { best = value; result = index; }
  });
  return { index: result, distance: best };
}

function pointSegmentDistance(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared ? Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared)) : 0;
  return Math.hypot(point.x - (a.x + t * dx), point.y - (a.y + t * dy));
}

function insertPoint(point) {
  if (points.length < 2) return false;
  let segment = 0;
  let best = Infinity;
  for (let index = 0; index < points.length - 1; index += 1) {
    const value = pointSegmentDistance(point, points[index], points[index + 1]);
    if (value < best) { best = value; segment = index; }
  }
  pushHistory();
  points.splice(segment + 1, 0, point);
  selectedPoint = segment + 1;
  updateTraceControls();
  draw();
  return true;
}

function pushHistory() {
  history.push(points.map((point) => ({ ...point })));
  if (history.length > 30) history.shift();
  $("#undo").disabled = false;
}

function perpendicularDistance(point, start, end) {
  return pointSegmentDistance(point, start, end);
}

function simplifyTrace(trace, epsilon = 1.5) {
  if (trace.length <= 2) return trace;
  let farthest = 0;
  let distanceMax = 0;
  for (let index = 1; index < trace.length - 1; index += 1) {
    const value = perpendicularDistance(trace[index], trace[0], trace[trace.length - 1]);
    if (value > distanceMax) { farthest = index; distanceMax = value; }
  }
  if (distanceMax <= epsilon) return [trace[0], trace[trace.length - 1]];
  const left = simplifyTrace(trace.slice(0, farthest + 1), epsilon);
  const right = simplifyTrace(trace.slice(farthest), epsilon);
  return left.slice(0, -1).concat(right);
}

canvas.addEventListener("pointerdown", (event) => {
  if (currentFrameIndex !== currentTask.frame_index || $("input[name='trace-state']:checked").value === "not_identifiable") return;
  const point = canvasPoint(event);
  canvas.setPointerCapture(event.pointerId);
  if (points.length < 2) {
    pushHistory();
    points = [point];
    drawing = true;
    selectedPoint = 0;
  } else {
    const nearest = nearestPoint(point);
    const threshold = 14 * canvas.width / canvas.getBoundingClientRect().width;
    if (event.shiftKey) {
      insertPoint(point);
      return;
    }
    if (nearest.distance <= threshold) {
      pushHistory();
      selectedPoint = nearest.index;
      dragging = true;
    } else {
      selectedPoint = -1;
    }
  }
  updateTraceControls();
  draw();
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing && !dragging) return;
  const point = canvasPoint(event);
  if (drawing) {
    const spacing = 3 * canvas.width / canvas.getBoundingClientRect().width;
    if (distance(points[points.length - 1], point) >= spacing) points.push(point);
    selectedPoint = points.length - 1;
  } else if (selectedPoint >= 0) {
    point.support = points[selectedPoint].support;
    points[selectedPoint] = point;
  }
  updateTraceControls();
  draw();
});

function finishPointer() {
  if (drawing) {
    points = simplifyTrace(points, 1.5);
    selectedPoint = points.length - 1;
  }
  drawing = false;
  dragging = false;
  updateTraceControls();
  draw();
}
canvas.addEventListener("pointerup", finishPointer);
canvas.addEventListener("pointercancel", finishPointer);

function updateTraceControls() {
  const notTraceable = $("input[name='trace-state']:checked").value === "not_identifiable";
  $("#trace-options").hidden = notTraceable;
  $("#point-count").textContent = points.length ? `${points.length} editable points` : "No trace yet";
  $("#reverse").disabled = points.length < 2;
  $("#undo").disabled = history.length === 0;
  $("#point-support").disabled = selectedPoint < 0 || notTraceable;
  if (selectedPoint >= 0 && points[selectedPoint]) $("#point-support").value = points[selectedPoint].support;
  $("#save-next").disabled = saving || (!notTraceable && points.length < 2);
  $("#canvas-hint").textContent = notTraceable
    ? "This frame will be retained as not identifiable; no fake centerline is required."
    : "Drag once from one end of the worm to the other. Then drag points to refine; Shift-click a segment to add a point.";
}

$("#retrace").addEventListener("click", () => {
  pushHistory();
  points = [];
  selectedPoint = -1;
  updateTraceControls();
  draw();
});

$("#reverse").addEventListener("click", () => {
  if (points.length < 2) return;
  pushHistory();
  points.reverse();
  const startOutside = $("#outside-start").checked;
  $("#outside-start").checked = $("#outside-end").checked;
  $("#outside-end").checked = startOutside;
  const orientation = $("input[name='head-tail']:checked").value;
  const reversed = orientation === "start_is_head" ? "start_is_tail" : orientation === "start_is_tail" ? "start_is_head" : "ambiguous";
  $(`input[name='head-tail'][value='${reversed}']`).checked = true;
  selectedPoint = selectedPoint >= 0 ? points.length - 1 - selectedPoint : -1;
  updateTraceControls();
  draw();
});

$("#undo").addEventListener("click", () => {
  if (!history.length) return;
  points = history.pop();
  selectedPoint = points.length ? points.length - 1 : -1;
  updateTraceControls();
  draw();
});

$("#point-support").addEventListener("change", (event) => {
  if (selectedPoint < 0) return;
  pushHistory();
  points[selectedPoint].support = event.target.value;
  draw();
});

$$("input[name='trace-state']").forEach((input) => input.addEventListener("change", () => { updateTraceControls(); draw(); }));
$("#zoom").addEventListener("input", () => { applyZoom(); draw(); });
$("#target-frame").addEventListener("click", () => loadFrame(currentTask.frame_index));

async function saveCurrent() {
  if (saving || $("#save-next").disabled) return;
  saving = true;
  updateTraceControls();
  setMessage("Saving…");
  const traceState = $("input[name='trace-state']:checked").value;
  const payload = {
    sample_id: currentTask.sample_id,
    annotation_pass: currentTask.annotation_pass,
    started_at_utc: startedAt,
    trace_state: traceState,
    head_tail_state: $("input[name='head-tail']:checked").value,
    outside_fov_at_start: $("#outside-start").checked,
    outside_fov_at_end: $("#outside-end").checked,
    worm_width_px: $("#worm-width").value || null,
    difficulty: $$("#difficulty-fields input:checked").map((input) => input.value),
    vertices: traceState === "not_identifiable" ? [] : points.map((point) => ({
      xy: [Number(point.x.toFixed(2)), Number(point.y.toFixed(2))],
      support_state: point.support,
    })),
  };
  try {
    const response = await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Save failed");
    await loadState();
  } catch (error) {
    setMessage(error.message);
  } finally {
    saving = false;
    updateTraceControls();
  }
}

$("#save-next").addEventListener("click", saveCurrent);

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    saveCurrent();
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    $("#undo").click();
  } else if (event.key === "0" && currentTask) {
    loadFrame(currentTask.frame_index);
  } else if ((event.key === "[" || event.key === "]") && currentTask) {
    const values = currentTask.temporal_window_indices;
    const position = values.indexOf(currentFrameIndex);
    const next = event.key === "[" ? Math.max(0, position - 1) : Math.min(values.length - 1, position + 1);
    loadFrame(values[next]);
  }
});

loadState().catch((error) => {
  workArea.hidden = true;
  emptyState.hidden = false;
  $("#empty-title").textContent = "Could not start the annotation tool";
  $("#empty-copy").textContent = error.message;
});
