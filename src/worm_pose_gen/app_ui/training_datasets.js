// Datasets tab: one row per dataset of the setup (Lab, or Mine with what it
// extends) with its labels by split, recordings, mask-only vs complete and
// readiness warnings. A row expands to its recordings; clicking one opens
// Labeling's Browse labels filtered to it (#labeling/browse/<dataset>/<recording>).
// Freeze benchmark freezes the dataset's spread-sampled test labels as a
// personal benchmark and queues every model's evaluation on it.

import {api, el, post} from "./api.js";

// Below these a dataset still trains, but its numbers are not worth much.
const FEW_TRAIN_LABELS = 20;
const FEW_TEST_LABELS = 10;

function readiness(summary) {
  const warnings = [...summary.readiness];
  if (summary.by_split.train && summary.by_split.train < FEW_TRAIN_LABELS) warnings.push(`only ${summary.by_split.train} training labels`);
  if (summary.by_split.test && summary.by_split.test < FEW_TEST_LABELS) warnings.push(`only ${summary.by_split.test} test labels`);
  return warnings;
}

function statusText(byStatus) {
  const parts = [];
  if (byStatus.complete) parts.push(`${byStatus.complete} complete`);
  if (byStatus.auto) parts.push(`${byStatus.auto} unreviewed body`);
  if (byStatus.mask_only) parts.push(`${byStatus.mask_only} mask only`);
  return parts.join(" · ") || "—";
}

function freezeDialog(ctx, page, summary) {
  return new Promise((resolve) => {
    const dialog = el("dialog", {class: "tr-dialog"});
    const description = el("input", {type: "text", placeholder: "optional, e.g. copper plates, first three recordings", "aria-label": "Description"});
    const error = el("p", {class: "note error"});
    const freeze = el("button", {class: "primary", onclick: async () => {
      freeze.disabled = true;
      try {
        const frozen = await post("/api/library/benchmarks", {dataset: summary.ref, description: description.value.trim()});
        const {queued} = await post("/api/training/evaluations", {setup: page.setup.ref, benchmark: frozen.ref});
        ctx.toast(`Froze ${frozen.ref} (${frozen.labels} labels); evaluating ${queued.length} models on it in the background`, "ok");
        dialog.close();
        resolve(frozen);
      } catch (failure) {
        error.textContent = failure.message;
        freeze.disabled = false;
      }
    }}, "Freeze benchmark");
    dialog.append(el("h2", {}, `Freeze a benchmark from ${summary.name}`),
      el("p", {class: "note"}, `The ${summary.by_split.test} test labels from ${summary.recordings_by_split.test} recordings `
        + "that were sampled to cover recordings (not fixes) become a personal benchmark. It never changes; freeze a new one when the test set has grown."),
      el("label", {}, "Description", description), error,
      el("div", {class: "row tr-dialog-buttons"}, el("button", {onclick: () => { dialog.close(); resolve(null); }}, "Cancel"), freeze));
    dialog.addEventListener("close", () => { dialog.remove(); resolve(null); });
    document.body.append(dialog);
    dialog.showModal();
  });
}

function recordingsTable(page, summary) {
  return el("table", {class: "data tr-recordings"},
    el("thead", {}, el("tr", {}, el("th", {}, "Recording"), el("th", {}, "Split"), el("th", {class: "num"}, "Labels"), el("th", {}, "Status"))),
    el("tbody", {}, summary.recordings.map((r) => el("tr", {},
      el("td", {}, el("button", {class: "link", title: "Browse these labels in Labeling",
        onclick: () => page.ctx.navigate("labeling", `browse/${summary.ref}/${r.recording}`)}, r.recording)),
      el("td", {}, el("span", {class: `badge tr-split-${r.split}`}, r.split || "—")),
      el("td", {class: "num"}, r.labels),
      el("td", {class: "note"}, statusText(r.statuses))))));
}

export async function render(container, page) {
  const {datasets} = await api(`/api/library/datasets?setup=${encodeURIComponent(page.setup.ref)}`);
  const {benchmarks} = await api(`/api/library/benchmarks?setup=${encodeURIComponent(page.setup.ref)}`);
  if (!datasets.length) {
    container.append(el("p", {class: "empty"}, "No dataset for this setup yet. Labels saved in Labeling start your dataset."));
    return;
  }
  const expanded = new Set();
  const table = el("table", {class: "data tr-datasets"});
  const draw = () => {
    const head = el("tr", {}, el("th", {}, ""), el("th", {}, "Dataset"), el("th", {}, "From"), el("th", {class: "num"}, "Train"),
      el("th", {class: "num"}, "Val"), el("th", {class: "num"}, "Test"), el("th", {class: "num"}, "Recordings"), el("th", {}, "Labels"),
      el("th", {}, "Readiness"), el("th", {}, ""));
    const rows = [];
    for (const s of datasets) {
      const open = expanded.has(s.ref);
      const warnings = readiness(s);
      const toggle = el("button", {class: "tr-expand", "aria-expanded": String(open), "aria-label": `Recordings of ${s.name}`,
        onclick: () => { open ? expanded.delete(s.ref) : expanded.add(s.ref); draw(); }}, open ? "▾" : "▸");
      const rec = s.recordings_by_split;
      rows.push(el("tr", {class: "tr-dataset", dataset: {ref: s.ref}},
        el("td", {}, toggle),
        el("td", {}, el("strong", {}, s.name), el("div", {class: "mp-sub"}, s.extends ? `${s.ref} · extends ${s.extends}` : s.ref)),
        el("td", {}, el("span", {class: `badge ${s.ref.startsWith("lab:") ? "mp-lab" : "mp-mine"}`}, s.ref.startsWith("lab:") ? "Lab" : "Mine")),
        el("td", {class: "num"}, s.by_split.train), el("td", {class: "num"}, s.by_split.val), el("td", {class: "num"}, s.by_split.test),
        el("td", {class: "num", title: `${rec.train} train · ${rec.val} val · ${rec.test} test`}, s.recordings.length),
        el("td", {class: "note"}, statusText(s.by_status)),
        el("td", {}, warnings.length ? warnings.map((w) => el("span", {class: "badge warn"}, w)) : el("span", {class: "badge ok"}, "ready")),
        el("td", {}, el("button", {disabled: !s.by_split.test, title: s.by_split.test ? "" : "no test labels",
          onclick: async () => { if (await freezeDialog(page.ctx, page, s)) page.go("datasets"); }}, "Freeze benchmark"))));
      if (open) rows.push(el("tr", {class: "tr-dataset-detail"}, el("td", {}), el("td", {colspan: 9}, recordingsTable(page, s))));
    }
    table.replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
  };
  draw();
  container.append(table,
    el("p", {class: "note tr-benchmarks"}, benchmarks.length
      ? `Benchmarks: ${benchmarks.map((b) => `${b.ref} (${b.labels} labels, ${b.recordings.length} recordings)`).join(", ")}.`
      : "No benchmark yet."));
}
