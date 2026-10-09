// Train tab: start a training job for the setup. Starting from a body-field
// model or from scratch trains a body-field net; starting from a mask-only
// segmenter fine-tunes the segmenter. The form is checked by the server as it
// changes (POST /api/training/plan: the kind, the label counts and the
// generated name), and Start submits one `train` job (POST /api/jobs), which
// prepares body targets, trains, evaluates on every benchmark and writes the
// card; its progress then shows at the top of the Models tab.

import {api, el, post} from "./api.js";
import {lagsText, loadModels} from "./model_picker.js";

const SCRATCH = "";
const KIND_TEXT = {
  body_net: "Trains a body-field net: mask, A-P field, head, tail and crossings.",
  segmenter: "Fine-tunes the mask-only segmenter: it gives the mask, no head/tail.",
};

function numberField(parameter, value, onInput) {
  const input = el("input", {type: "number", name: parameter.name, value, step: parameter.type === "int" ? 1 : "any",
    min: parameter.min, max: parameter.max, oninput: onInput});
  return el("label", {class: "tr-field", title: parameter.help || ""}, parameter.label, input);
}

function runOnFields(compute) {
  const local = compute?.local?.available, slurm = compute?.slurm?.available;
  const choice = el("select", {name: "run_on", "aria-label": "Run on"},
    local ? el("option", {value: "local", selected: compute.default_run_on === "local"},
      `This machine (${compute.local.gpus.length} GPU${compute.local.gpus.length === 1 ? "" : "s"})`) : null,
    slurm ? el("option", {value: "slurm", selected: compute.default_run_on === "slurm"}, "SLURM") : null);
  const defaults = compute?.slurm?.defaults || {};
  const partition = el("select", {name: "partition", "aria-label": "SLURM partition"},
    el("option", {value: ""}, "cluster default"),
    (compute?.slurm?.partitions || []).map((p) => el("option", {value: p.name, selected: p.name === defaults.partition}, p.name)));
  const time = el("input", {type: "text", name: "time", value: defaults.time || "01:00:00", "aria-label": "SLURM time limit"});
  const slurmFields = el("div", {class: "row tr-slurm"}, el("label", {class: "tr-field"}, "Partition", partition),
    el("label", {class: "tr-field"}, "Time limit", time));
  const update = () => { slurmFields.hidden = choice.value !== "slurm"; };
  choice.addEventListener("change", update);
  update();
  const wrapper = el("div", {class: "tr-runon", hidden: !(local && slurm)}, el("label", {class: "tr-field"}, "Run on", choice), slurmFields);
  return {
    element: wrapper,
    can: !!(local || slurm),
    reason: compute ? compute.reason : "could not read where jobs run",
    value: () => ({run_on: choice.value, slurm: choice.value === "slurm" ? {partition: partition.value || null, time: time.value.trim()} : null}),
  };
}

export async function render(container, page, params) {
  const {ctx} = page;
  const setup = page.setup;
  const [models, {datasets}, schema] = await Promise.all([
    loadModels(setup.ref), api(`/api/library/datasets?setup=${encodeURIComponent(setup.ref)}`), api("/api/training/schema"),
  ]);
  const byRef = Object.fromEntries(models.models.map((m) => [m.ref, m]));
  const requested = decodeURIComponent(params || "");
  const start = byRef[requested] ? requested : (byRef[setup.defaults?.body] ? setup.defaults.body : (byRef[setup.defaults?.mask] ? setup.defaults.mask : SCRATCH));

  // ----- start from
  const startSelect = el("select", {name: "start_from", "aria-label": "Start from"},
    models.models.map((m) => el("option", {value: m.ref, selected: m.ref === start},
      `${m.name} (${m.scope === "lab" ? "Lab" : "Mine"}, ${m.kind === "body_net" ? "body-field net" : "segmenter"}${m.default_for.length ? `, ★ ${m.default_for.join(" + ")}` : ""})`)),
    el("option", {value: SCRATCH, selected: start === SCRATCH}, "From scratch (ImageNet weights)"));
  const kindNote = el("p", {class: "note tr-kind"});
  const contextRadios = Object.entries(schema.contexts).map(([key, lags]) => el("label", {class: "inline"},
    el("input", {type: "radio", name: "context", value: key, checked: key === "none"}),
    key === "none" ? "None (single frame)" : `Short: ±${lags.join(", ")} frames${setup.fps ? ` (${lags.map((l) => +(l / setup.fps).toFixed(2)).join(", ")} s)` : ""}`));
  const contextRow = el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Temporal context"), el("div", {class: "row"}, contextRadios));

  // ----- training data: one dataset; its recordings' splits decide what trains and validates
  const counts = (d) => `${d.ref.startsWith("lab:") ? "Lab" : "Mine"} · ${d.by_split.train} train / ${d.by_split.val} val labels`;
  const preferred = datasets.find((d) => d.ref.startsWith("mine:")) || datasets[0];
  const dataSelect = el("select", {name: "dataset", "aria-label": "Dataset", onchange: () => check()},
    datasets.map((d) => el("option", {value: d.ref, selected: d === preferred}, `${d.name} (${counts(d)})`)));
  const dataList = datasets.length ? dataSelect
    : el("p", {class: "note"}, "No dataset for this setup yet: make one in the Datasets tab and choose its recordings' splits.");

  // ----- settings
  let kind = (byRef[start]?.kind === "segmenter") ? "segmenter" : "body_net";
  const edited = new Set();
  const fields = {};
  const shown = el("div", {class: "tr-fields"});
  const advanced = el("div", {class: "tr-fields"});
  const defaultOf = (p) => (typeof p.default === "object" ? p.default[kind] : p.default);
  const drawFields = () => {
    const appliesTo = (p) => !p.kinds || p.kinds.includes(kind);
    shown.replaceChildren();
    advanced.replaceChildren();
    for (const p of schema.parameters.filter(appliesTo)) {
      const value = edited.has(p.name) && fields[p.name] ? fields[p.name].value : defaultOf(p);
      const field = numberField(p, value, () => { edited.add(p.name); check(); });
      fields[p.name] = field.querySelector("input");
      (p.advanced ? advanced : shown).append(field);
    }
  };

  const name = el("input", {type: "text", name: "name", "aria-label": "Name", oninput: () => { name.dataset.edited = "1"; check(); }});
  const notes = el("textarea", {name: "notes", rows: 2, "aria-label": "Notes", placeholder: "what this run tries; goes on the model card"});
  const runOn = runOnFields(page.compute);
  const summary = el("p", {class: "note tr-plan"});
  const startButton = el("button", {class: "primary", type: "submit"}, "Start training");

  const request = () => {
    const values = {};
    for (const [key, input] of Object.entries(fields)) if (input.isConnected && input.value !== "") values[key] = Number(input.value);
    return {
      setup: setup.ref, dataset: datasets.length ? dataSelect.value : "", start_from: startSelect.value || null,
      context: startSelect.value ? "none" : (container.querySelector('input[name="context"]:checked')?.value || "none"),
      params: values, name: name.dataset.edited ? name.value.trim() : "", notes: notes.value.trim(),
    };
  };

  let checking = 0, timer = null, planOk = false;
  async function check() {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const token = ++checking;
      try {
        const plan = await post("/api/training/plan", request());
        if (token !== checking) return;
        planOk = true;
        if (!name.dataset.edited) name.value = plan.name;
        summary.className = "note tr-plan";
        summary.textContent = `${plan.train} training and ${plan.val} validation labels from ${plan.recordings} recordings`
          + (plan.mask_only ? ` (${plan.mask_only} train the mask only)` : "") + ".";
      } catch (error) {
        if (token !== checking) return;
        planOk = false;
        summary.className = "note error tr-plan";
        summary.textContent = error.message;
      }
      startButton.disabled = !planOk || !runOn.can;
    }, 200);
  }

  const onStart = () => {
    const ref = startSelect.value;
    const model = byRef[ref];
    const next = model?.kind === "segmenter" ? "segmenter" : "body_net";
    if (next !== kind) { kind = next; drawFields(); }
    contextRow.hidden = !!ref;
    kindNote.textContent = `${KIND_TEXT[kind]} ${model ? `Keeps its inputs: ${lagsText(model.inputs)}.` : "Starts from ImageNet weights."}`;
    check();
  };
  startSelect.addEventListener("change", onStart);
  for (const radio of contextRadios) radio.addEventListener("change", check);

  const form = el("form", {class: "tr-form", onsubmit: async (event) => {
    event.preventDefault();
    startButton.disabled = true;
    try {
      const job = await post("/api/jobs", {kind: "train", ...request(), ...runOn.value()});
      ctx.toast(`Started ${job.spec.params.name}`, "ok");
      page.go("models");
    } catch (error) {
      summary.className = "note error tr-plan";
      summary.textContent = error.message;
      startButton.disabled = false;
    }
  }},
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Start from"), el("div", {}, startSelect, kindNote)),
    contextRow,
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Training data"), el("div", {}, dataList, summary)),
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Settings"), el("div", {}, shown,
      el("details", {class: "tr-advanced"}, el("summary", {}, "Advanced"), advanced))),
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Name"), el("div", {}, name)),
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Notes"), el("div", {}, notes)),
    runOn.element.hidden ? null : el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, ""), runOn.element),
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, ""), el("div", {class: "row"}, startButton,
      runOn.can ? null : el("span", {class: "note error"}, `Training is unavailable: ${runOn.reason}`))),
  );
  drawFields();
  onStart();
  container.append(form);
  return () => clearTimeout(timer);
}
