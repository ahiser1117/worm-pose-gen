// Analyse: which models and where the job runs (docs/APP_SIMPLIFICATION.md,
// sections 2 and 4). The Recordings screen analyses with the setup's
// default models in one click; "Change" (there, and "Change model" in a
// workspace's header) opens the dialog below. Run on is one choice built
// from GET /api/compute: hidden when only one place can run jobs, and
// Analyse is disabled with the reason when none can. Developers also get
// the stage checklist, the GPU and, for a new workspace, a frame range.

import {api, el} from "./api.js";
import {openModelPicker} from "./model_picker.js";

const STORE_KEY = "workspace.runOn";

function read(key, fallback) {
  try { const raw = localStorage.getItem(key); return raw === null ? fallback : JSON.parse(raw); } catch { return fallback; }
}
function write(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage blocked */ }
}

let computePromise = null;
export function loadCompute() {
  computePromise ??= api("/api/compute").catch((error) => { computePromise = null; throw error; });
  return computePromise;
}

// Where analysis jobs run: the user's last choice when it is still offered, else the server's default.
export class RunOn {
  constructor(compute) {
    this.compute = compute;
    this.options = [];
    if (compute.local.available) this.options.push({value: "local", label: `This machine (${compute.local.gpus.length} GPU${compute.local.gpus.length === 1 ? "" : "s"})`});
    if (compute.slurm.available) this.options.push({value: "slurm", label: "SLURM cluster"});
    const saved = read(STORE_KEY, {});
    this.value = this.options.some((o) => o.value === saved.run_on) ? saved.run_on : compute.default_run_on;
    this.slurm = {partition: saved.partition ?? compute.slurm.defaults?.partition ?? "", time: saved.time ?? compute.slurm.defaults?.time ?? ""};
  }

  get canRun() { return !!this.compute.can_run; }
  get reason() { return this.compute.reason || "No GPU or SLURM is available for jobs."; }

  // The placement fields of a job request.
  payload() {
    if (this.value !== "slurm") return {run_on: this.value};
    return {run_on: "slurm", slurm: {partition: this.slurm.partition || null, time: this.slurm.time || null}};
  }

  save() { write(STORE_KEY, {run_on: this.value, ...this.slurm}); }

  // The control, or null when there is nothing to choose.
  control(onchange = () => {}) {
    if (this.options.length < 2) return null;
    const select = el("select", {"aria-label": "Run on"}, this.options.map((o) => el("option", {value: o.value, selected: o.value === this.value}, o.label)));
    const partition = el("input", {type: "text", value: this.slurm.partition, placeholder: "default", "aria-label": "SLURM partition"});
    const time = el("input", {type: "text", value: this.slurm.time, placeholder: "1:00:00", "aria-label": "SLURM time limit"});
    const slurmFields = el("span", {class: "ws-slurm"}, el("label", {class: "inline"}, "partition", partition), el("label", {class: "inline"}, "time", time));
    const sync = () => { slurmFields.hidden = this.value !== "slurm"; };
    select.addEventListener("change", () => { this.value = select.value; sync(); this.save(); onchange(); });
    partition.addEventListener("change", () => { this.slurm.partition = partition.value.trim(); this.save(); });
    time.addEventListener("change", () => { this.slurm.time = time.value.trim(); this.save(); });
    sync();
    return el("span", {class: "ws-run-on"}, el("label", {class: "inline"}, "Run on", select), slurmFields);
  }
}

// Model cards of a setup by reference (for names), cached per setup.
const cards = new Map();
export async function modelCards(setup) {
  if (!cards.has(setup)) {
    cards.set(setup, api(`/api/library/models?setup=${encodeURIComponent(setup)}`).then(({models}) => new Map(models.map((m) => [m.ref, m]))).catch((error) => {
      cards.delete(setup);
      throw error;
    }));
  }
  return cards.get(setup);
}

export function modelLabel(models) {
  const names = [models.mask?.name, models.body?.name].filter(Boolean);
  return names.length ? names.join(" + ") : "no model";
}

// The dialog: resolves {models: {mask, body} (refs), stages?, gpu?, first?, last?, step?} or null.
//   options: {title, setup, models: {mask: {ref, name}, body}, runOn (RunOn | null),
//             submitLabel, warning, dev, newWorkspace (frames count for the range fields)}
export function openAnalyseDialog(ctx, options) {
  const {title, setup, runOn = null, submitLabel = "Done", warning = null, dev = false, newWorkspace = null} = options;
  const chosen = {mask: options.models.mask ? {...options.models.mask} : null, body: options.models.body ? {...options.models.body} : null};
  return new Promise((resolve) => {
    let result = null;
    const dialog = el("dialog", {class: "ws-dialog ws-analyse"});
    const modelRow = (role, label) => {
      const name = el("span", {class: "ws-model-name"});
      const render = () => { name.textContent = chosen[role]?.name || (role === "body" ? "none (orientation from the body taper only)" : "none"); };
      render();
      const change = el("button", {type: "button", onclick: async () => {
        const ref = await openModelPicker(ctx, {setup, role, current: chosen[role]?.ref ?? null});
        if (!ref) return;
        const card = (await modelCards(setup)).get(ref);
        chosen[role] = {ref, name: card?.name || ref};
        render();
      }}, "Change");
      const none = role === "body" ? el("button", {type: "button", class: "link", onclick: () => { chosen.body = null; render(); }}, "None") : null;
      return el("div", {class: "ws-form-row"}, el("span", {class: "ws-form-label"}, label), name, change, none);
    };
    const rows = [modelRow("mask", "Mask model"), modelRow("body", "Body model")];
    const runControl = runOn?.control();
    if (runControl) rows.push(el("div", {class: "ws-form-row"}, runControl));
    let devFields = null;
    if (dev) devFields = devOptions(runOn, newWorkspace);
    const submit = el("button", {class: "primary", type: "button", disabled: runOn && !runOn.canRun, title: runOn && !runOn.canRun ? runOn.reason : null, onclick: () => {
      result = {models: {mask: chosen.mask?.ref ?? null, body: chosen.body?.ref ?? null}, mask: chosen.mask, body: chosen.body, ...(devFields?.values() || {})};
      dialog.close();
    }}, submitLabel);
    dialog.append(
      el("h2", {}, title), ...rows,
      devFields?.node ?? null,
      warning ? el("p", {class: "note warn"}, warning) : null,
      runOn && !runOn.canRun ? el("p", {class: "note error"}, runOn.reason) : null,
      el("div", {class: "ws-dialog-actions"}, el("button", {type: "button", onclick: () => dialog.close()}, "Cancel"), submit),
    );
    dialog.addEventListener("close", () => { dialog.remove(); resolve(result); });
    document.body.append(dialog);
    dialog.showModal();
  });
}

const DEV_STAGES = ["segment", "prior", "fit", "ambiguity", "propagate", "track", "fixed_body"];

function devOptions(runOn, newWorkspace) {
  const boxes = DEV_STAGES.map((stage) => el("input", {type: "checkbox", value: stage, checked: stage !== "fixed_body"}));
  const gpus = runOn?.compute.local.gpus || [];
  const gpu = el("select", {"aria-label": "GPU"}, el("option", {value: ""}, "Automatic"), gpus.map((g) => el("option", {value: g.index}, `GPU ${g.index}${g.name ? ` · ${g.name}` : ""}`)));
  const range = newWorkspace ? ["first", "last", "step"].map((key) => el("input", {type: "number", min: key === "step" ? 1 : 0, placeholder: key === "first" ? "0" : key === "last" ? String(newWorkspace - 1) : "1", "aria-label": key})) : [];
  const node = el("fieldset", {class: "ws-dev-fields"}, el("legend", {}, "Developer"),
    el("div", {class: "ws-form-row ws-stages"}, el("span", {class: "ws-form-label"}, "Stages"), boxes.map((box) => el("label", {class: "inline"}, box, box.value))),
    gpus.length ? el("div", {class: "ws-form-row"}, el("span", {class: "ws-form-label"}, "GPU"), gpu) : null,
    range.length ? el("div", {class: "ws-form-row"}, el("span", {class: "ws-form-label"}, "Frames"), range.map((input) => el("label", {class: "inline"}, input.getAttribute("aria-label"), input))) : null,
  );
  return {
    node,
    values() {
      const values = {stages: boxes.filter((b) => b.checked).map((b) => b.value)};
      if (gpu.value !== "") values.gpu = Number(gpu.value);
      for (const input of range) if (input.value !== "") values[input.getAttribute("aria-label")] = Number(input.value);
      return values;
    },
  };
}
