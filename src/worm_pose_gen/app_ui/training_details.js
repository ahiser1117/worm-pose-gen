// A model's Details (Models tab): the loss curves, the worst benchmark frames
// as overlays, the settings, the labels it was trained on and its notes, from
// GET /api/training/models/<ref>.

import {api, el} from "./api.js";
import {fmt, lagsText} from "./model_picker.js";
import {chartLegend, lossChart} from "./training_charts.js";

const SETTING_LABELS = {
  max_epochs: "Max epochs", epochs: "Max epochs", learning_rate: "Learning rate", batch_size: "Batch size", crop_size: "Crop size",
  patience: "Patience", plateau_patience: "Plateau patience", encoder_lr_scale: "Encoder LR scale", ap_weight: "A-P loss weight",
  heatmap_weight: "Head/tail loss weight", overlap_weight: "Overlap loss weight", min_fit_iou: "Minimum body fit IoU", seed: "Seed",
  lags: "Lags (frames)", context: "Temporal context",
};

function section(title, ...children) {
  return el("section", {class: "tr-detail-section"}, el("h3", {}, title), ...children);
}

function worstFrames(evaluations) {
  const refs = Object.keys(evaluations);
  if (!refs.length) return el("p", {class: "note"}, "Not evaluated on any benchmark yet.");
  const grid = el("div", {class: "tr-worst"});
  const summary = el("p", {class: "note"});
  const show = (ref) => {
    const e = evaluations[ref];
    summary.textContent = `${ref}: IoU mean ${fmt.iou(e.iou_mean)}, worst 5% ${fmt.iou(e.iou_worst5)} (${e.worst5_count} of ${e.labels} labels), `
      + `head/tail ${fmt.pct(e.head_tail_correct)}, A-P error ${fmt.err(e.ap_error)}`;
    grid.replaceChildren(...e.worst.map((w) => el("figure", {},
      el("img", {src: w.url, alt: `${w.recording} frame ${w.frame}`, loading: "lazy"}),
      el("figcaption", {}, `${w.recording} · ${w.frame} · IoU ${fmt.iou(w.iou)}${w.head_tail_correct === false ? " · head/tail wrong" : ""}`))));
  };
  const select = refs.length > 1
    ? el("select", {"aria-label": "Benchmark", onchange: () => show(select.value)}, refs.map((r) => el("option", {value: r}, r)))
    : null;
  show(refs[0]);
  return el("div", {}, select ? el("div", {class: "row"}, select) : null, summary,
    el("p", {class: "note"}, "Magenta: worm the model missed; green: extra; red ring: predicted head; blue: tail."), grid);
}

function settings(card) {
  const rows = Object.entries(card.hparams || {}).filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => el("tr", {}, el("th", {}, SETTING_LABELS[k] || k), el("td", {}, Array.isArray(v) ? (v.join(", ") || "none") : String(v))));
  const inputs = card.inputs || {};
  rows.unshift(
    el("tr", {}, el("th", {}, "Inputs"), el("td", {}, `${lagsText(inputs)}; ${inputs.preprocessing || ""}${inputs.fps ? `; ${inputs.fps} fps` : ""}${inputs.pixel_size_um ? `; ${(+inputs.pixel_size_um).toFixed(3)} µm/px` : ""}`)),
    el("tr", {}, el("th", {}, "Outputs"), el("td", {}, card.outputs.join(", "))),
    el("tr", {}, el("th", {}, "Started from"), el("td", {}, card.parent || "ImageNet weights")),
  );
  return el("table", {class: "data tr-settings"}, el("tbody", {}, rows));
}

function labelsUsed(details) {
  if (!details.labels.length) {
    const entries = details.card.trained_on || [];
    return el("p", {class: "note"}, entries.length
      ? entries.map((e) => `${e.dataset}: ${e.counts?.train ?? 0} train, ${e.counts?.val ?? 0} val labels from ${e.recordings} recordings (fingerprint ${e.fingerprint})`).join("; ")
      : "No record of the labels used.");
  }
  return el("div", {},
    el("p", {class: "note"}, `${details.label_count} label revisions of ${(details.card.trained_on || []).map((e) => e.dataset).join(", ") || "the dataset"}; `
      + "the exact list is in training/labels.json."),
    el("table", {class: "data"},
      el("thead", {}, el("tr", {}, el("th", {}, "Recording"), el("th", {class: "num"}, "Train"), el("th", {class: "num"}, "Val"))),
      el("tbody", {}, details.labels.map((r) => el("tr", {}, el("td", {}, r.recording),
        el("td", {class: "num"}, r.train), el("td", {class: "num"}, r.val))))));
}

export async function openDetails(ctx, ref) {
  let details;
  try {
    details = await api(`/api/training/models/${ref}`);
  } catch (error) {
    ctx.toast(error.message, "error");
    return;
  }
  const {card} = details;
  const run = details.run || {};
  const facts = [card.kind === "body_net" ? "body-field net" : "segmenter", card.author && `by ${card.author}`,
    card.created_at && card.created_at.slice(0, 10), run.best_epoch && `kept epoch ${run.best_epoch} of ${run.epochs_run}`].filter(Boolean);
  const dialog = el("dialog", {class: "tr-dialog tr-details", "aria-label": `Details of ${card.name}`});
  dialog.append(...[
    el("div", {class: "tr-details-head"}, el("h2", {}, card.name), el("span", {class: "note"}, `${ref} · ${facts.join(" · ")}`),
      el("button", {class: "tr-close", "aria-label": "Close", onclick: () => dialog.close()}, "Close")),
    details.missing.length ? el("p", {class: "note warn"}, details.missing.join("; ")) : null,
    el("div", {class: "tr-details-grid"},
      section("Loss", details.curve.length ? el("div", {}, lossChart(details.curve, {width: 420, height: 160}), chartLegend())
        : el("p", {class: "note"}, "No training curve recorded.")),
      section("Settings", settings(card))),
    section("Worst benchmark frames", worstFrames(details.evaluations)),
    section("Labels used", labelsUsed(details)),
    section("Notes", el("p", {class: "tr-notes"}, card.notes || "—"),
      details.defaults_log.length ? el("ul", {class: "note"}, details.defaults_log.map((e) =>
        el("li", {}, `${e.at.slice(0, 10)}: made the ${e.role} default by ${e.who} — ${e.reason}`))) : null),
  ].filter(Boolean));
  dialog.addEventListener("close", () => dialog.remove());
  document.body.append(dialog);
  dialog.showModal();
}
