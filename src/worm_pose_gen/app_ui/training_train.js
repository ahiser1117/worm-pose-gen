// Train tab: start a training job for the setup. Starting from a body-field
// model or from scratch trains a body-field net; starting from a mask-only
// segmenter fine-tunes the segmenter. The form starts from a model's recipe
// (model_training.recipe): a fine-tune takes the chosen model's settings, and
// from scratch takes the setup's default body model's context, encoder,
// pseudo-label sets and fibre share, so a run trains the way that model was
// trained, on the labels of the chosen dataset. The form is checked by the server as it
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
  const [models, {datasets}, schema, {pseudo_labels: pseudoSets}] = await Promise.all([
    loadModels(setup.ref), api(`/api/library/datasets?setup=${encodeURIComponent(setup.ref)}`), api("/api/training/schema"),
    api(`/api/library/pseudo-labels?setup=${encodeURIComponent(setup.ref)}`),
  ]);
  const byRef = Object.fromEntries(models.models.map((m) => [m.ref, m]));
  const requested = decodeURIComponent(params || "");
  const start = byRef[requested] ? requested : (byRef[setup.defaults?.body] ? setup.defaults.body : (byRef[setup.defaults?.mask] ? setup.defaults.mask : SCRATCH));

  // ----- start from
  const startSelect = el("select", {name: "start_from", "aria-label": "Start from"},
    models.models.map((m) => el("option", {value: m.ref, selected: m.ref === start},
      `${m.name} (${m.scope === "lab" ? "Lab" : "Mine"}, ${m.kind === "body_net" ? "body-field net" : "segmenter"}${m.default_for.length ? `, ★ ${m.default_for.join(" + ")}` : ""})`)),
    el("option", {value: SCRATCH, selected: start === SCRATCH}, "From scratch (pretrained encoder)"));
  const kindNote = el("p", {class: "note tr-kind"});
  const contextRadios = Object.entries(schema.contexts).map(([key, lags]) => el("label", {class: "inline"},
    el("input", {type: "radio", name: "context", value: key, checked: key === "none"}),
    key === "none" ? "None (single frame)" : `Short: ±${lags.join(", ")} frames${setup.fps ? ` (${lags.map((l) => +(l / setup.fps).toFixed(2)).join(", ")} s)` : ""}`));
  const contextRow = el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Temporal context"), el("div", {class: "row"}, contextRadios));
  const encoderRadios = Object.entries(schema.encoders).map(([key, text]) => el("label", {class: "inline"},
    el("input", {type: "radio", name: "encoder", value: key, checked: key === "resnet18"}), text));
  const encoderRow = el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Encoder"), el("div", {class: "row"}, encoderRadios));

  // ----- training data: one dataset; its recordings' splits decide what trains and validates
  const counts = (d) => `${d.ref.startsWith("lab:") ? "Lab" : "Mine"} · ${d.by_split.train} train / ${d.by_split.val} val labels`;
  const preferred = datasets.find((d) => d.ref.startsWith("mine:")) || datasets[0];
  const dataSelect = el("select", {name: "dataset", "aria-label": "Dataset", onchange: () => check()},
    datasets.map((d) => el("option", {value: d.ref, selected: d === preferred}, `${d.name} (${counts(d)})`)));
  const dataList = datasets.length ? dataSelect
    : el("p", {class: "note"}, "No dataset for this setup yet: make one in the Datasets tab and choose its recordings' splits.");

  // ----- extra data: pseudo-label sets and fibre distractors (body-field nets only)
  const scope = (ref) => (ref.startsWith("lab:") ? "Lab" : "Mine");
  const pseudoBoxes = pseudoSets.map((set) => el("label", {class: "inline", title: set.description || ""},
    el("input", {type: "checkbox", name: "pseudo_labels", value: set.ref, onchange: () => check()}),
    `${set.name} (${scope(set.ref)} · ${set.items} items from ${set.recordings} recordings)`));
  const shareField = (name, label, help) => {
    const input = el("input", {type: "number", name, value: 0, step: "any", min: 0, max: 1, oninput: () => check()});
    return {input, element: el("label", {class: "tr-field", title: help}, label, input)};
  };
  const pseudoShare = shareField("pseudo_fraction", "Pseudo-label share", "share of every epoch drawn from the checked pseudo-label sets");
  const fiberShare = shareField("fiber_share", "Fibre share", "share of hand labels given fibre distractors on the agar");
  const recipeNote = el("p", {class: "note"});
  const recipeRow = el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Extra data"), el("div", {},
    pseudoSets.length ? el("div", {class: "row"}, pseudoBoxes) : el("p", {class: "note"}, "No pseudo-label sets for this setup."),
    el("div", {class: "tr-fields"}, pseudoSets.length ? pseudoShare.element : null, fiberShare.element), recipeNote));

  // ----- settings
  let kind = (byRef[start]?.kind === "segmenter") ? "segmenter" : "body_net";
  const edited = new Set();
  let recipeParams = {};
  const fields = {};
  const shown = el("div", {class: "tr-fields"});
  const advanced = el("div", {class: "tr-fields"});
  const defaultOf = (p) => (typeof p.default === "object" ? p.default[kind] : p.default);
  const drawFields = () => {
    const appliesTo = (p) => !p.kinds || p.kinds.includes(kind);
    shown.replaceChildren();
    advanced.replaceChildren();
    for (const p of schema.parameters.filter(appliesTo)) {
      const value = edited.has(p.name) && fields[p.name] ? fields[p.name].value : (recipeParams[p.name] ?? defaultOf(p));
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
    const body = kind === "body_net";
    const pseudo = body ? pseudoBoxes.map((box) => box.querySelector("input")).filter((input) => input.checked).map((input) => input.value) : [];
    return {
      setup: setup.ref, dataset: datasets.length ? dataSelect.value : "", start_from: startSelect.value || null,
      context: startSelect.value ? "none" : (container.querySelector('input[name="context"]:checked')?.value || "none"),
      encoder: startSelect.value ? "resnet18" : (container.querySelector('input[name="encoder"]:checked')?.value || "resnet18"),
      params: values, pseudo_labels: pseudo, pseudo_fraction: pseudo.length ? Number(pseudoShare.input.value) : 0,
      fiber_share: body ? Number(fiberShare.input.value) : 0, name: name.dataset.edited ? name.value.trim() : "", notes: notes.value.trim(),
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

  // A fine-tune repeats its model's recipe; from scratch repeats the default body model's, without its learning settings.
  const applyRecipe = (model, fineTune) => {
    const recipe = model?.recipe;
    recipeParams = fineTune && recipe ? recipe.params : {};
    edited.clear();
    drawFields();
    const sets = new Set(recipe?.pseudo_labels || []);
    for (const box of pseudoBoxes) { const input = box.querySelector("input"); input.checked = sets.has(input.value); }
    pseudoShare.input.value = recipe?.pseudo_fraction || 0;
    fiberShare.input.value = recipe?.fiber_share || 0;
    if (!fineTune) {
      for (const radio of contextRadios) radio.querySelector("input").checked = radio.querySelector("input").value === (recipe?.context || "none");
      for (const radio of encoderRadios) radio.querySelector("input").checked = radio.querySelector("input").value === (recipe?.encoder || "resnet18");
    }
    const missing = [...sets].filter((ref) => !pseudoSets.some((set) => set.ref === ref));
    recipeNote.className = missing.length ? "note error" : "note";
    recipeNote.textContent = !recipe ? ""
      : `${fineTune ? "Settings" : "Context, encoder and extra data"} as ${model.name} was trained.`
        + (missing.length ? ` Its pseudo-labels ${missing.join(", ")} are not in a library you can read.` : "");
  };

  const onStart = () => {
    const ref = startSelect.value;
    const model = byRef[ref];
    kind = model?.kind === "segmenter" ? "segmenter" : "body_net";
    applyRecipe(model || byRef[setup.defaults?.body], !!model);
    contextRow.hidden = !!ref;
    encoderRow.hidden = !!ref;
    recipeRow.hidden = kind !== "body_net";
    kindNote.textContent = `${KIND_TEXT[kind]} ${model ? `Keeps its inputs: ${lagsText(model.inputs)}.` : "Starts from the encoder's pretrained weights."}`;
    check();
  };
  startSelect.addEventListener("change", onStart);
  for (const radio of [...contextRadios, ...encoderRadios]) radio.addEventListener("change", check);

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
    encoderRow,
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Training data"), el("div", {}, dataList, summary)),
    recipeRow,
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Settings"), el("div", {}, shown,
      el("details", {class: "tr-advanced"}, el("summary", {}, "Advanced"), advanced))),
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Name"), el("div", {}, name)),
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, "Notes"), el("div", {}, notes)),
    runOn.element.hidden ? null : el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, ""), runOn.element),
    el("div", {class: "tr-form-row"}, el("span", {class: "tr-form-label"}, ""), el("div", {class: "row"}, startButton,
      runOn.can ? null : el("span", {class: "note error"}, `Training is unavailable: ${runOn.reason}`))),
  );
  onStart();
  container.append(form);
  return () => clearTimeout(timer);
}
