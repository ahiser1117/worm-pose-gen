// The model picker: one table of a setup's models with their inputs, outputs
// and benchmark numbers (docs/APP_SIMPLIFICATION.md, section 4). The Training
// page's Models tab shows the same table with row actions; the Workspace
// opens it as a dialog for "Change model":
//
//   const ref = await openModelPicker(ctx, {setup: "lab:nir-flv", role: "mask", current: "lab:nir-hand284"});
//   // ref: the chosen model ref, or null when cancelled
//
// Rows come from GET /api/training/models?setup=&benchmark= (lab and personal
// models mixed, ★ the setup's default per role). A model with no evaluation
// on the chosen benchmark shows "Not evaluated" and its evaluation is queued
// in the background (POST /api/training/evaluations); the table polls until
// the numbers arrive. A model missing an output a stage uses says what it
// cannot do. For a setup without models the table lists the lab models of
// the other setups with their numbers there.

import {api, el, post, query} from "./api.js";

const ROLE_LABEL = {mask: "mask", body: "body"};
const KIND_LABEL = {segmenter: "segmenter", body_net: "body-field net"};
const POLL_MS = 3000;
// (setup, benchmark) pairs whose missing evaluations were already requested in this page session.
const requested = new Set();

export const fmt = {
  iou: (v) => (v === null || v === undefined ? "—" : v.toFixed(3)),
  pct: (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`),
  err: (v) => (v === null || v === undefined ? "—" : v.toFixed(3)),
};

export function lagsText(inputs) {
  const frames = inputs?.lags_frames || [];
  if (!frames.length) return "frame";
  const seconds = inputs.lags_s ? ` (${inputs.lags_s.map((s) => +s.toFixed(2)).join("/")} s)` : "";
  return `frame + motion ±${frames.join("/")} fr${seconds}`;
}

export function trainedText(row) {
  const t = row.trained_on || {};
  return t.labels ? `${t.labels} labels · ${t.recordings} rec` : "—";
}

export async function loadModels(setup, benchmark = "") {
  return api(`/api/training/models${query({setup, benchmark})}`);
}

// Queue the evaluations the table is missing, once per (setup, benchmark).
export async function fillEvaluations(data) {
  const benchmark = data.benchmark;
  if (!benchmark || data.others) return false;
  const missing = data.models.some((row) => !row.evaluation && !row.evaluating);
  const key = `${data.setup.ref} ${benchmark}`;
  if (!missing || requested.has(key)) return false;
  requested.add(key);
  await post("/api/training/evaluations", {setup: data.setup.ref, benchmark});
  return true;
}

export function pendingEvaluations(data) {
  return data.models.some((row) => row.evaluating || (!row.evaluation && data.benchmark && !data.others));
}

function benchmarkLabel(summary) {
  return `${summary.ref} · ${plural(summary.labels, "label")}`;
}

// The benchmark selector shown above the table.
export function benchmarkSelect(data, onChange) {
  if (!data.benchmarks.length) {
    return el("span", {class: "note"}, data.others ? "" : "No benchmark for this setup yet: freeze one in Datasets.");
  }
  const select = el("select", {class: "mp-benchmark", "aria-label": "Benchmark", onchange: () => onChange(select.value)},
    data.benchmarks.map((b) => el("option", {value: b.ref, selected: b.ref === data.benchmark}, benchmarkLabel(b))));
  return el("label", {class: "inline mp-benchmark-label"}, "Benchmark", select);
}

// The best value of a metric column, when the models differ in it (it is then highlighted).
function best(rows, key, lower = false) {
  const values = rows.map((r) => r.evaluation?.[key]).filter((v) => v !== null && v !== undefined);
  if (values.length < 2 || values.every((v) => v === values[0])) return null;
  return lower ? Math.min(...values) : Math.max(...values);
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function metricCells(row, bests) {
  const e = row.evaluation;
  if (!e) {
    const text = row.evaluating
      ? (row.evaluating.state === "running" ? `Evaluating… ${Math.round(100 * (row.evaluating.progress || 0))}%` : "Evaluation queued")
      : (row.benchmark ? "Not evaluated" : "No benchmark");
    return [el("td", {class: "num mp-pending", colspan: 4}, text)];
  }
  const cell = (key, text, lower = false) => el("td", {
    class: `num${bests[key] !== null && e[key] === bests[key] ? " mp-best" : ""}`,
  }, text);
  const worst = el("td", {class: `num${bests.iou_worst5 !== null && e.iou_worst5 === bests.iou_worst5 ? " mp-best" : ""}`,
    title: `mean IoU of the worst ${e.worst5_count} of ${e.labels} labels`}, fmt.iou(e.iou_worst5));
  return [
    cell("iou_mean", fmt.iou(e.iou_mean)), worst,
    el("td", {class: "num", title: e.head_tail_labels ? `${e.head_tail_labels} labels with both ends in view` : ""}, fmt.pct(e.head_tail_correct)),
    cell("ap_error", fmt.err(e.ap_error), true),
  ];
}

// The table: mode "page" adds the row actions (onAction(name, row)); mode
// "picker" makes rows choosable for `role` (onPick(row)).
export function modelTable(data, {mode = "page", role = null, current = null, onPick = null, onAction = null} = {}) {
  // In the picker, the models that can fill the role come first, its default at the top.
  const rows = !role ? data.models : [...data.models].sort((a, b) =>
    (b.roles.includes(role) - a.roles.includes(role)) || (b.default_for.includes(role) - a.default_for.includes(role)));
  if (!rows.length) return el("p", {class: "empty"}, "No models for this setup yet. Train one, or ask for a lab model.");
  const bests = {iou_mean: best(rows, "iou_mean"), iou_worst5: best(rows, "iou_worst5"), ap_error: best(rows, "ap_error", true)};
  const head = el("tr", {},
    el("th", {}, "Model"), el("th", {}, "From"), el("th", {}, "Inputs"), el("th", {}, "Outputs"), el("th", {}, "Trained on"),
    el("th", {class: "num"}, "IoU mean"), el("th", {class: "num"}, "IoU worst 5%"), el("th", {class: "num"}, "Head/tail"),
    el("th", {class: "num"}, "A-P err"), el("th", {}, ""));
  const body = rows.map((row) => {
    const fits = !role || row.roles.includes(role);
    const stars = row.default_for.map((r) => el("span", {class: "mp-star", title: `default ${ROLE_LABEL[r]} model`}, `★ ${ROLE_LABEL[r]}`));
    const sub = [KIND_LABEL[row.kind] || row.kind];
    if (data.others) sub.push(`on ${row.setup_name || row.setup}`);
    if (row.parent) sub.push(`from ${row.parent.split(":")[1]}`);
    const actions = [];
    if (mode === "page") {
      actions.push(
        el("button", {class: "link", onclick: () => onAction("default", row)}, "Use as default"),
        el("button", {class: "link", onclick: () => onAction("train", row)}, "Train from this"),
        el("button", {class: "link", onclick: () => onAction("details", row)}, "Details"),
      );
    } else {
      actions.push(row.ref === current
        ? el("span", {class: "badge ok"}, "in use")
        : el("button", {class: fits ? "primary" : "", disabled: !fits, onclick: () => onPick(row),
          title: fits ? "" : `has no ${role} output`}, "Use"));
    }
    const tr = el("tr", {
      class: `mp-row${row.ref === current ? " mp-current" : ""}${fits ? "" : " mp-unfit"}`, dataset: {ref: row.ref},
    },
      el("td", {}, el("div", {class: "mp-name"}, el("strong", {}, row.name), ...stars), el("div", {class: "mp-sub"}, sub.join(" · "))),
      el("td", {}, el("span", {class: `badge ${row.scope === "lab" ? "mp-lab" : "mp-mine"}`}, row.scope === "lab" ? "Lab" : "Mine")),
      el("td", {class: "mp-inputs"}, lagsText(row.inputs)),
      el("td", {}, el("div", {class: "mp-chips"}, row.outputs.map((o) => el("span", {class: "mp-chip"}, o === "ap" ? "A-P" : o))),
        row.missing.length ? el("div", {class: "mp-missing"}, row.missing[0]) : null),
      el("td", {}, trainedText(row)),
      ...metricCells(row, bests),
      el("td", {}, el("div", {class: "mp-actions"}, actions)),
    );
    if (row.missing.length > 1) tr.querySelector(".mp-missing").title = row.missing.join("\n");
    if (mode === "picker" && fits && row.ref !== current) tr.addEventListener("dblclick", () => onPick(row));
    return tr;
  });
  const table = el("table", {class: "data mp-table"}, el("thead", {}, head), el("tbody", {}, body));
  const notes = [];
  const evaluated = rows.find((r) => r.evaluation);
  if (evaluated && evaluated.evaluation.labels < 20) {
    const e = evaluated.evaluation;
    notes.push(`${data.benchmark} has ${plural(e.labels, "label")}: its worst 5% is the worst ${plural(e.worst5_count, "label")}.`);
  }
  if (data.others) notes.unshift("This setup has no models yet. These are the lab models of other microscopes, with their numbers there: pick the closest one as the default.");
  return el("div", {class: "mp-wrap"}, notes.map((n) => el("p", {class: "note"}, n)), el("div", {class: "mp-scroll"}, table));
}

// The picker dialog: resolves to the chosen ref, or null.
export async function openModelPicker(ctx, {setup, role, current = null}) {
  let data = await loadModels(setup);
  return new Promise((resolve) => {
    let timer = null, done = false;
    const dialog = el("dialog", {class: "model-picker", "aria-label": "Choose a model"});
    const finish = (value) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      if (dialog.open) dialog.close();
      dialog.remove();
      resolve(value);
    };
    const body = el("div", {class: "mp-dialog-body"});
    const render = () => {
      body.replaceChildren(
        el("div", {class: "row mp-toolbar"}, benchmarkSelect(data, async (benchmark) => { data = await loadModels(setup, benchmark); refresh(); })),
        modelTable(data, {mode: "picker", role, current, onPick: (row) => finish(row.ref)}),
      );
    };
    const refresh = async () => {
      render();
      clearTimeout(timer);
      try { await fillEvaluations(data); } catch (error) { ctx.toast(`Could not start the evaluations: ${error.message}`, "error"); }
      if (pendingEvaluations(data)) {
        timer = setTimeout(async () => {
          if (done) return;
          try { data = await loadModels(setup, data.benchmark); } catch { /* keep the last table */ }
          refresh();
        }, POLL_MS);
      }
    };
    dialog.append(
      el("h2", {}, `Choose the ${ROLE_LABEL[role] || ""} model`.replace("  ", " ")),
      el("p", {class: "note"}, role === "body"
        ? "The body model gives the A-P field and head/tail evidence."
        : "The mask model segments the worm in every frame."),
      body,
      el("div", {class: "row mp-footer"}, el("button", {onclick: () => finish(null)}, "Cancel")),
    );
    dialog.addEventListener("close", () => finish(null));
    document.body.append(dialog);
    dialog.showModal();
    refresh();
  });
}
