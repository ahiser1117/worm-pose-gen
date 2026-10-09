// Models tab: the setup's training runs (running with a live loss curve and
// Cancel; failed with their error until dismissed), then the model picker
// table with Use as default, Train from this and Details.

import {api, el, post} from "./api.js";
import {benchmarkSelect, fillEvaluations, loadModels, modelTable, pendingEvaluations} from "./model_picker.js";
import {chartLegend, lossChart} from "./training_charts.js";
import {openDetails} from "./training_details.js";

const POLL_MS = 2000;
const KIND_LABEL = {segmenter: "segmenter fine-tune", body_net: "body-field net"};

function runCard(run, refresh, ctx) {
  const params = run.params || {};
  const result = run.result || {};
  const what = [KIND_LABEL[params.kind] || params.kind, params.start_from ? `from ${params.start_from.split(":")[1]}` : "from scratch",
    `${params.train} train · ${params.val} val labels`].join(" · ");
  if (run.state === "failed") {
    return el("div", {class: "tr-run failed", dataset: {job: run.id}},
      el("div", {class: "tr-run-head"}, el("strong", {}, `Training ${params.name} failed`), el("span", {class: "note"}, what),
        el("button", {class: "tr-run-action", onclick: async () => { await post(`/api/training/runs/${run.id}/dismiss`); refresh(); }}, "Dismiss")),
      el("pre", {class: "tr-error"}, (run.error || "no error message").split("\n").slice(-8).join("\n")));
  }
  const phase = {prepare: "Preparing body targets", train: "Training", evaluate: "Evaluating", done: "Writing the card"}[result.phase] || "Queued";
  const epoch = result.epoch ? `epoch ${result.epoch} of at most ${result.max_epochs}` : "";
  const best = result.best_val_loss !== undefined && result.best_val_loss !== null ? `best validation loss ${result.best_val_loss.toFixed(4)}` : "";
  return el("div", {class: "tr-run", dataset: {job: run.id}},
    el("div", {class: "tr-run-head"},
      el("strong", {}, `${run.state === "queued" ? "Queued" : phase}: ${params.name}`), el("span", {class: "note"}, what),
      el("button", {class: "tr-run-action danger", onclick: async (event) => {
        event.target.disabled = true;
        try { await post(`/api/jobs/${run.id}/cancel`); ctx.toast(`Cancelled ${params.name}`); } catch (error) { ctx.toast(error.message, "error"); }
        refresh();
      }}, "Cancel")),
    el("progress", {max: 1, value: run.progress || 0}),
    el("div", {class: "tr-run-body"},
      el("div", {class: "tr-run-chart"}, lossChart(result.curve || [], {width: 320, height: 110, maxEpochs: result.max_epochs}), chartLegend()),
      el("div", {class: "tr-run-facts note"}, [epoch, best, run.message].filter(Boolean).map((t) => el("div", {}, t)))));
}

function defaultDialog(ctx, page, row) {
  return new Promise((resolve) => {
    const roles = row.roles;
    const dialog = el("dialog", {class: "tr-dialog"});
    const reason = el("input", {type: "text", placeholder: "Why? e.g. better heads on copper plates", "aria-label": "Reason", required: true});
    // Preselect the first role the model is not the default for yet.
    const preferred = roles.find((r) => !row.default_for.includes(r)) || roles[0];
    const choices = roles.map((role) => {
      const current = page.setup.defaults?.[role];
      return el("label", {class: "inline tr-role"}, el("input", {type: "radio", name: "role", value: role, checked: role === preferred}),
        `${role} model`, el("span", {class: "note"}, current ? `now ${current}` : "none yet"));
    });
    const error = el("p", {class: "note error"});
    const save = el("button", {class: "primary", onclick: async () => {
      const role = dialog.querySelector('input[name="role"]:checked')?.value;
      if (!role) { error.textContent = "Choose the role."; return; }
      if (!reason.value.trim()) { error.textContent = "Give a one-line reason; it is logged with the change."; reason.focus(); return; }
      try {
        await post(`/api/library/setups/${page.setup.ref}/defaults`, {role, model: row.ref, reason: reason.value.trim()});
        ctx.toast(`${row.name} is now the ${role} default for ${page.setup.name}`, "ok");
        dialog.close();
        resolve(true);
      } catch (failure) { error.textContent = failure.message; }
    }}, "Use as default");
    dialog.append(el("h2", {}, `Use ${row.name} as default`),
      el("p", {class: "note"}, `For ${page.setup.name}. The Workspace analyses new recordings with the defaults.`),
      roles.length ? el("div", {class: "stack"}, choices) : el("p", {class: "note error"}, "This model has no output a default needs."),
      el("label", {}, "Reason", reason), error,
      el("div", {class: "row tr-dialog-buttons"}, el("button", {onclick: () => { dialog.close(); resolve(false); }}, "Cancel"), save));
    dialog.addEventListener("close", () => { dialog.remove(); resolve(false); });
    document.body.append(dialog);
    dialog.showModal();
    reason.focus();
  });
}

export async function render(container, page, params) {
  const {ctx} = page;
  const runs = el("div", {class: "tr-runs"});
  const toolbar = el("div", {class: "row tr-toolbar"});
  const table = el("div", {class: "tr-models"});
  container.append(runs, toolbar, table);
  let data = null, benchmark = "", timer = null, stopped = false, active = false;

  const onAction = async (name, row) => {
    if (name === "details") openDetails(ctx, row.ref);
    else if (name === "train") page.go("train", row.ref);
    else if (name === "default" && await defaultDialog(ctx, page, row)) page.go("models");
  };

  async function refresh() {
    clearTimeout(timer);
    if (stopped) return;
    let shown = [];
    try {
      [data, {runs: shown}] = await Promise.all([loadModels(page.setup.ref, benchmark), api(`/api/training/runs?setup=${encodeURIComponent(page.setup.ref)}`)]);
    } catch (error) {
      table.replaceChildren(el("p", {class: "note error"}, error.message));
      return;
    }
    if (stopped) return;
    active = shown.some((r) => r.state === "queued" || r.state === "running");
    runs.replaceChildren(...shown.map((run) => runCard(run, refresh, ctx)));
    toolbar.replaceChildren(benchmarkSelect(data, (value) => { benchmark = value; refresh(); }),
      el("span", {class: "spacer"}), el("button", {onclick: () => page.go("train")}, "Train a model"));
    table.replaceChildren(modelTable(data, {mode: "page", onAction}));
    try { await fillEvaluations(data); } catch (error) { ctx.toast(`Could not start the evaluations: ${error.message}`, "error"); }
    if (active || pendingEvaluations(data)) timer = setTimeout(refresh, POLL_MS);
  }

  await refresh();
  return () => { stopped = true; clearTimeout(timer); };
}
