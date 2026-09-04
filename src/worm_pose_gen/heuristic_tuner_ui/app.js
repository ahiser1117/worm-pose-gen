"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const state = {
  defaults: null,
  latest: null,
  stage: "candidate",
  timer: null,
  request: null,
};

const fields = {
  local_radius: $("#local-radius"),
  smooth_radius: $("#smooth-radius"),
  foreground_z: $("#foreground-z"),
  connected_foreground_z: $("#connected-foreground-z"),
  close_radius: $("#close-radius"),
};

const outputs = {
  local_radius: $("#local-radius-value"),
  smooth_radius: $("#smooth-radius-value"),
  foreground_z: $("#foreground-z-value"),
  connected_foreground_z: $("#connected-foreground-z-value"),
  close_radius: $("#close-radius-value"),
};

function formatValue(name, value) {
  return name.endsWith("_z") ? `${Number(value).toFixed(2)} z` : `${Number(value)} px`;
}

function updateLabels() {
  Object.entries(fields).forEach(([name, input]) => {
    outputs[name].textContent = formatValue(name, input.value);
  });
  const enabled = $("#connected-enabled").checked;
  fields.connected_foreground_z.disabled = !enabled;
  $("#connected-cutoff-field").classList.toggle("disabled", !enabled);
}

function currentPayload() {
  return {
    sample_id: $("#sample-select").value,
    local_radius: Number(fields.local_radius.value),
    smooth_radius: Number(fields.smooth_radius.value),
    foreground_z: Number(fields.foreground_z.value),
    connected_enabled: $("#connected-enabled").checked,
    connected_foreground_z: Number(fields.connected_foreground_z.value),
    close_radius: Number(fields.close_radius.value),
  };
}

function keepCutoffsOrdered(changedName) {
  const high = Number(fields.foreground_z.value);
  const low = Number(fields.connected_foreground_z.value);
  if (low < high) return;
  if (changedName === "foreground_z") {
    fields.connected_foreground_z.value = Math.max(0.25, high - 0.05).toFixed(2);
  } else {
    fields.foreground_z.value = Math.min(6, low + 0.05).toFixed(2);
  }
}

function scheduleAnalysis(changedName = "") {
  keepCutoffsOrdered(changedName);
  updateLabels();
  clearTimeout(state.timer);
  state.timer = setTimeout(runAnalysis, 180);
}

function setStatus(message, error = false) {
  $("#run-status").textContent = message;
  $("#run-status").style.color = error ? "var(--red)" : "";
}

function number(value) {
  return Number(value).toLocaleString();
}

function updateMetrics(metrics) {
  $("#metric-retained").textContent = `${number(metrics.retained_component_area)} px²`;
  $("#metric-recovered").textContent = metrics.recovered_area
    ? `${number(metrics.recovered_area)} px² (+${metrics.recovered_percent.toFixed(1)}%)`
    : "off / none";
  $("#metric-components").textContent = number(metrics.high_component_count);
  $("#metric-disconnected").textContent = `${number(metrics.disconnected_faint_area)} px²`;
}

const legends = {
  score: [
    ["linear-gradient(90deg,#32125b,#922363,#ee5b2b,#fcf4c2)", "low → high local darkness"],
  ],
  candidate: [
    ["var(--magenta)", "primary cutoff"],
    ["var(--amber)", "faint cutoff candidate"],
  ],
  kept: [
    ["var(--accent)", "high-confidence body"],
    ["var(--green)", "connected faint recovery"],
    ["var(--red)", "discarded high object"],
  ],
};

function renderStage() {
  if (!state.latest) return;
  $$(".stage-tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.stage === state.stage);
  });
  const analysis = $("#analysis-image");
  analysis.src = state.latest.images[state.stage];
  const opacity = Number($("#overlay-opacity").value);
  analysis.style.opacity = state.stage === "score" ? "1" : String(opacity);
  $("#base-image").style.opacity = state.stage === "score" ? "0" : "1";
  $("#overlay-opacity").disabled = state.stage === "score";
  const legend = $("#legend");
  legend.replaceChildren();
  legends[state.stage].forEach(([color, label]) => {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = color;
    item.append(swatch, document.createTextNode(label));
    legend.append(item);
  });
}

async function runAnalysis() {
  if (!$("#sample-select").value) return;
  if (state.request) state.request.abort();
  const controller = new AbortController();
  state.request = controller;
  $("#loading-overlay").hidden = false;
  setStatus("Recomputing score and connectivity…");
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentPayload()),
      signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Analysis failed");
    state.latest = payload;
    $("#base-image").src = payload.images.frame;
    updateMetrics(payload.metrics);
    renderStage();
    $("#download-config").disabled = false;
    setStatus(`Updated · score range ${payload.metrics.score_min.toFixed(2)} to ${payload.metrics.score_max.toFixed(2)} z`);
  } catch (error) {
    if (error.name !== "AbortError") setStatus(error.message, true);
  } finally {
    if (state.request === controller) {
      state.request = null;
      $("#loading-overlay").hidden = true;
    }
  }
}

function applyDefaults() {
  Object.entries(fields).forEach(([name, input]) => {
    let value = state.defaults[name];
    if (name === "connected_foreground_z" && value === null) {
      // Hysteresis is disabled by default; seed the slider at half the primary cutoff.
      value = Math.max(0.25, state.defaults.foreground_z / 2).toFixed(2);
    }
    input.value = value;
  });
  $("#connected-enabled").checked = state.defaults.connected_foreground_z !== null;
  scheduleAnalysis();
}

function downloadConfig() {
  if (!state.latest) return;
  const content = JSON.stringify(state.latest.config, null, 2) + "\n";
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([content], { type: "application/json" }));
  link.download = "local-darkness-config.json";
  link.click();
  URL.revokeObjectURL(link.href);
}

async function initialize() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load the frame catalog");
    const payload = await response.json();
    state.defaults = payload.defaults;
    const select = $("#sample-select");
    payload.samples.forEach((sample) => {
      const option = document.createElement("option");
      option.value = sample.sample_id;
      option.textContent = sample.label;
      select.append(option);
    });
    select.value = payload.default_sample_id;
    applyDefaults();
  } catch (error) {
    setStatus(error.message, true);
    $("#loading-overlay").textContent = error.message;
  }
}

Object.entries(fields).forEach(([name, input]) => {
  input.addEventListener("input", () => scheduleAnalysis(name));
});
$("#connected-enabled").addEventListener("change", () => scheduleAnalysis());
$("#sample-select").addEventListener("change", () => scheduleAnalysis());
$("#reset-config").addEventListener("click", applyDefaults);
$("#download-config").addEventListener("click", downloadConfig);
$("#overlay-opacity").addEventListener("input", renderStage);
$$(".stage-tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    state.stage = button.dataset.stage;
    renderStage();
  });
});

initialize();
