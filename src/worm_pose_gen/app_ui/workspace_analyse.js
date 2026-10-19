// Analyse: which models, which pipeline settings and where the job runs
// (docs/APP_SIMPLIFICATION.md, sections 2 and 4). The form below sits in the
// Issues panel of a workspace not analysed yet (the setup's default models
// first) and in the Re-analyse dialog (the workspace's last choices). With a
// body model it also asks where the masks come from: the mask model (the
// default) or the body model's own mask output, which then needs no mask
// model. The pipeline settings are the server's short list (fitting schedule,
// mask threshold, prior, tracking), folded, each starting from the value last
// used. Run on is one choice built from GET /api/compute: hidden when only
// one place can run jobs, and Analyse is disabled with the reason when none
// can. Developers also get the stage checklist and the GPU.

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

export function modelLabel(models, maskSource = "segmenter") {
  if (maskSource === "body_net" && models.body) return `${models.body.name} (masks and body)`;
  const names = [models.mask?.name, models.body?.name].filter(Boolean);
  return names.length ? names.join(" + ") : "no model";
}

// The form: the models, Masks from, Run on, the pipeline settings (folded, with how many
// differ from their defaults) and, for developers, the stages and the GPU.
//   options: {setup, models: {mask: {ref, name}, body}, maskSource ("segmenter" | "body_net"),
//             runOn (RunOn), settings (GET /api/workspaces/<name>/pipeline-settings),
//             values ({key: value}, the settings last used), dev, compact (narrow, for a side panel)}
// values() is the Analyse request: {models: {mask, body} (refs), mask_source, pipeline (the
// settings that differ from their defaults), run_on, slurm?, stages?, gpu?}.
export class AnalyseForm {
  constructor(ctx, options) {
    const {setup, runOn, settings = [], values = {}, dev = false, compact = false} = options;
    this.runOn = runOn;
    this.chosen = {mask: options.models.mask ? {...options.models.mask} : null, body: options.models.body ? {...options.models.body} : null};
    this.maskSource = options.maskSource === "body_net" ? "body_net" : "segmenter";
    const chosen = this.chosen;
    // Masks from: a choice only while there is a body model to take them from.
    const source = el("select", {"aria-label": "Masks from", onchange: () => { this.maskSource = source.value; }},
      el("option", {value: "segmenter"}, "Mask model"), el("option", {value: "body_net"}, "Body model"));
    const sourceRow = el("div", {class: "ws-form-row ws-mask-source"}, el("span", {class: "ws-form-label"}, "Masks from"), source);
    const renderSource = () => {
      if (!chosen.body) this.maskSource = "segmenter";
      source.value = this.maskSource;
      sourceRow.hidden = !chosen.body;
    };
    const modelRow = (role, label) => {
      const name = el("span", {class: "ws-model-name"});
      const render = () => { name.textContent = chosen[role]?.name || (role === "body" ? "none (orientation from the body taper only)" : "none"); renderSource(); };
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
    const runControl = runOn.control();
    this.pipeline = pipelineFields(settings, values);
    this.devFields = dev ? devOptions(runOn) : null;
    this.node = el("div", {class: `ws-analyse-form${compact ? " compact" : ""}`},
      modelRow("mask", "Mask model"), modelRow("body", "Body model"), sourceRow,
      runControl ? el("div", {class: "ws-form-row"}, runControl) : null,
      this.pipeline.node,
      this.devFields?.node ?? null,
      runOn.canRun ? null : el("p", {class: "note error"}, runOn.reason));
  }

  get canRun() { return this.runOn.canRun; }

  values() {
    return {
      models: {mask: this.chosen.mask?.ref ?? null, body: this.chosen.body?.ref ?? null}, mask_source: this.maskSource,
      pipeline: this.pipeline.values(), ...this.runOn.payload(), ...(this.devFields?.values() || {}),
    };
  }
}

// The dialog around the form: resolves the form's values() or null.
//   options: the form's, and {title, submitLabel, warning}
export function openAnalyseDialog(ctx, options) {
  const {title, submitLabel = "Analyse", warning = null} = options;
  const form = new AnalyseForm(ctx, options);
  return new Promise((resolve) => {
    let result = null;
    const dialog = el("dialog", {class: "ws-dialog ws-analyse"});
    const submit = el("button", {class: "primary", type: "button", disabled: !form.canRun, title: form.canRun ? null : form.runOn.reason, onclick: () => {
      result = form.values();
      dialog.close();
    }}, submitLabel);
    dialog.append(
      el("h2", {}, title), form.node,
      warning ? el("p", {class: "note warn"}, warning) : null,
      el("div", {class: "ws-dialog-actions"}, el("button", {type: "button", onclick: () => dialog.close()}, "Cancel"), submit),
    );
    dialog.addEventListener("close", () => { dialog.remove(); resolve(result); });
    document.body.append(dialog);
    dialog.showModal();
  });
}

// The pipeline settings, folded: one control per setting, starting from the values last used
// (else the default); values() gives the ones that differ from their defaults.
function pipelineFields(settings, last) {
  const value = (setting) => (setting.key in last ? last[setting.key] : setting.default);
  const fields = settings.map((setting) => {
    let input;
    if (setting.choices) input = el("select", {}, setting.choices.map((choice) => el("option", {value: choice, selected: choice === value(setting)}, choice)));
    else if (setting.type === "bool") input = el("input", {type: "checkbox", checked: !!value(setting)});
    else input = el("input", {type: "number", min: setting.min, max: setting.max, step: setting.type === "int" ? 1 : 0.05, value: value(setting)});
    input.setAttribute("aria-label", setting.label);
    input.dataset.key = setting.key;
    const read = () => {
      if (setting.type === "bool") return input.checked;
      if (setting.type === "str") return input.value;
      const number = Number(input.value);
      return input.value === "" || !Number.isFinite(number) ? setting.default : number;
    };
    const set = (v) => { if (setting.type === "bool") input.checked = !!v; else input.value = v; };
    const row = el("label", {class: "ws-setting", title: setting.help}, el("span", {}, setting.label), input);
    return {setting, input, row, read, set};
  });
  const count = el("span", {class: "note ws-pipeline-count"});
  const changed = () => fields.filter((f) => f.read() !== f.setting.default);
  const update = () => {
    const n = changed().length;
    count.textContent = n ? `${n} changed` : "defaults";
  };
  const reset = el("button", {type: "button", class: "link", onclick: () => { for (const f of fields) f.set(f.setting.default); update(); }}, "Reset to defaults");
  for (const f of fields) f.input.addEventListener("input", update);
  for (const f of fields) f.input.addEventListener("change", update);
  update();
  const node = settings.length
    ? el("details", {class: "ws-pipeline", open: changed().length > 0},
      el("summary", {}, "Pipeline settings ", count),
      fields.map((f) => f.row), el("div", {class: "ws-pipeline-reset"}, reset))
    : null;
  return {node, values: () => Object.fromEntries(changed().map((f) => [f.setting.key, f.read()]))};
}

const DEV_STAGES = ["segment", "prior", "fit", "ambiguity", "propagate", "track", "fixed_body"];

function devOptions(runOn) {
  const boxes = DEV_STAGES.map((stage) => el("input", {type: "checkbox", value: stage, checked: stage !== "fixed_body"}));
  const gpus = runOn?.compute.local.gpus || [];
  const gpu = el("select", {"aria-label": "GPU"}, el("option", {value: ""}, "Automatic"), gpus.map((g) => el("option", {value: g.index}, `GPU ${g.index}${g.name ? ` · ${g.name}` : ""}`)));
  const node = el("fieldset", {class: "ws-dev-fields"}, el("legend", {}, "Developer"),
    el("div", {class: "ws-form-row ws-stages"}, el("span", {class: "ws-form-label"}, "Stages"), boxes.map((box) => el("label", {class: "inline"}, box, box.value))),
    gpus.length ? el("div", {class: "ws-form-row"}, el("span", {class: "ws-form-label"}, "GPU"), gpu) : null,
  );
  return {
    node,
    values() {
      const values = {stages: boxes.filter((b) => b.checked).map((b) => b.value)};
      if (gpu.value !== "") values.gpu = Number(gpu.value);
      return values;
    },
  };
}
