// Datasets tab (#training/datasets[/<dataset ref>]): the setup's label
// collection (every recording with a label) as one table, and for the chosen
// dataset the split each recording's labels go to: Train, Val, Test or Not
// included. A recording is not included until someone chooses its split, so
// a new dataset includes nothing and a newly labeled recording joins no
// dataset by itself. The totals by split update with every choice. Lab
// datasets are read-only; New dataset makes a personal one, empty or with
// another dataset's splits. Clicking a recording opens Labeling's Browse
// labels with the dataset's splits (#labeling/browse/<dataset>/<recording>).
// Freeze benchmark freezes the dataset's spread-sampled test labels as a
// personal benchmark and queues every model's evaluation on it.

import {api, el, post} from "./api.js";

// Below these a dataset still trains, but its numbers are not worth much.
const FEW_TRAIN_LABELS = 20;
const FEW_TEST_LABELS = 10;
const SPLITS = [["train", "Train"], ["val", "Val"], ["test", "Test"]];
const NOT_INCLUDED = "";

function statusText(byStatus) {
  const parts = [];
  if (byStatus.complete) parts.push(`${byStatus.complete} complete`);
  if (byStatus.auto) parts.push(`${byStatus.auto} unreviewed body`);
  if (byStatus.mask_only) parts.push(`${byStatus.mask_only} mask only`);
  return parts.join(" · ") || "—";
}

// Labels and recordings by split (and not included), from the recording rows as they stand.
function totals(rows) {
  const result = Object.fromEntries([...SPLITS.map(([key]) => key), "none"].map((key) => [key, {labels: 0, recordings: 0}]));
  for (const row of rows) {
    const bucket = result[row.split || "none"];
    bucket.labels += row.labels;
    bucket.recordings += 1;
  }
  return result;
}

function readiness(counts) {
  const warnings = [];
  if (!counts.train.labels) warnings.push("no training labels");
  else if (counts.train.labels < FEW_TRAIN_LABELS) warnings.push(`only ${counts.train.labels} training labels`);
  if (!counts.val.recordings) warnings.push("no validation recording");
  if (!counts.test.recordings) warnings.push("no test recording");
  else if (counts.test.labels < FEW_TEST_LABELS) warnings.push(`only ${counts.test.labels} test labels`);
  return warnings;
}

function slug(text) {
  return text.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
}

function dialog(title, body, action, label) {
  return new Promise((resolve) => {
    const box = el("dialog", {class: "tr-dialog"});
    const error = el("p", {class: "note error"});
    const go = el("button", {class: "primary", onclick: async () => {
      go.disabled = true;
      try {
        resolve(await action());
        box.close();
      } catch (failure) {
        error.textContent = failure.message;
        go.disabled = false;
      }
    }}, label);
    box.append(el("h2", {}, title), ...body, error,
      el("div", {class: "row tr-dialog-buttons"}, el("button", {onclick: () => { box.close(); resolve(null); }}, "Cancel"), go));
    box.addEventListener("close", () => { box.remove(); resolve(null); });
    document.body.append(box);
    box.showModal();
  });
}

function newDatasetDialog(page, datasets) {
  const name = el("input", {type: "text", placeholder: "e.g. copper plates", "aria-label": "Name"});
  const from = el("select", {"aria-label": "Start from"},
    el("option", {value: ""}, "Nothing: every recording not included"),
    datasets.map((d) => el("option", {value: d.ref}, `The splits of ${d.name} (${d.ref})`)));
  return dialog("New dataset", [
    el("p", {class: "note"}, `A dataset of ${page.setup.name} chooses which of its labeled recordings train, validate or test a model.`),
    el("label", {}, "Name", name), el("label", {}, "Start from", from),
  ], async () => {
    const id = slug(name.value);
    if (!id) throw new Error("give the dataset a name");
    const source = datasets.find((d) => d.ref === from.value);
    const splits = source ? Object.fromEntries(source.recordings.filter((r) => r.split).map((r) => [r.recording, r.split])) : {};
    return post("/api/library/datasets", {id, setup: page.setup.ref, name: name.value.trim(), splits});
  }, "Create");
}

function freezeDialog(ctx, page, summary, counts) {
  const description = el("input", {type: "text", placeholder: "optional, e.g. copper plates, first three recordings", "aria-label": "Description"});
  return dialog(`Freeze a benchmark from ${summary.name}`, [
    el("p", {class: "note"}, `The ${counts.test.labels} test labels from ${counts.test.recordings} recordings `
      + "that were sampled to cover recordings (not fixes) become a personal benchmark. It never changes; freeze a new one when the test set has grown."),
    el("label", {}, "Description", description),
  ], async () => {
    const frozen = await post("/api/library/benchmarks", {dataset: summary.ref, description: description.value.trim()});
    const {queued} = await post("/api/training/evaluations", {setup: page.setup.ref, benchmark: frozen.ref});
    ctx.toast(`Froze ${frozen.ref} (${frozen.labels} labels); evaluating ${queued.length} models on it in the background`, "ok");
    return frozen;
  }, "Freeze benchmark");
}

export async function render(container, page, params) {
  const [{datasets}, {benchmarks}, collection] = await Promise.all([
    api(`/api/library/datasets?setup=${encodeURIComponent(page.setup.ref)}`),
    api(`/api/library/benchmarks?setup=${encodeURIComponent(page.setup.ref)}`),
    api(`/api/library/setups/${encodeURIComponent(page.setup.ref)}/collection`),
  ]);
  const requested = decodeURIComponent(params || "");
  let summary = datasets.find((d) => d.ref === requested) || datasets.find((d) => d.ref.startsWith("mine:")) || datasets[0] || null;

  const create = el("button", {onclick: async () => {
    const made = await newDatasetDialog(page, datasets);
    if (made) page.go("datasets", encodeURIComponent(made.ref));
  }}, "New dataset…");
  const picker = el("select", {class: "tr-dataset-select", "aria-label": "Dataset", onchange: () => page.go("datasets", encodeURIComponent(picker.value))},
    datasets.map((d) => el("option", {value: d.ref, selected: d === summary}, `${d.name} (${d.ref.startsWith("lab:") ? "Lab" : "Mine"}, ${d.ref})`)));
  container.append(el("div", {class: "row tr-dataset-bar"}, datasets.length ? el("label", {class: "inline"}, "Dataset", picker) : null, create),
    el("p", {class: "note"}, `${collection.labels} labels from ${collection.recordings.length} recordings in ${page.setup.name}. `
      + "Labels saved in Labeling join this collection; each dataset chooses which recordings train, validate or test."));
  if (!summary) {
    container.append(el("p", {class: "empty"}, collection.labels
      ? "No dataset for this setup yet. Make one to choose which recordings train, validate and test."
      : "No labels for this setup yet. Label frames in Labeling, then make a dataset here."));
    return;
  }

  const rows = summary.recordings;
  const totalsBar = el("div", {class: "tr-split-totals", "aria-live": "polite"});
  const freeze = el("button", {onclick: async () => { if (await freezeDialog(page.ctx, page, summary, totals(rows))) page.go("datasets", encodeURIComponent(summary.ref)); }},
    "Freeze benchmark");
  const drawTotals = () => {
    const counts = totals(rows);
    const tile = (key, label) => el("div", {class: `tr-split-total tr-split-${key}`},
      el("span", {class: "tr-split-name"}, label), el("strong", {}, `${counts[key].labels}`),
      el("span", {class: "note"}, `labels · ${counts[key].recordings} recording${counts[key].recordings === 1 ? "" : "s"}`));
    const warnings = readiness(counts);
    totalsBar.replaceChildren(...SPLITS.map(([key, label]) => tile(key, label)), tile("none", "Not included"),
      el("div", {class: "tr-readiness"}, warnings.length ? warnings.map((w) => el("span", {class: "badge warn"}, w)) : el("span", {class: "badge ok"}, "ready")));
    freeze.disabled = !counts.test.labels;
    freeze.title = counts.test.labels ? "" : "no test labels";
  };

  // A choice shows at once; the server's answer then replaces the rows, and a refusal puts the old split back.
  const choose = async (row, select) => {
    const previous = row.split;
    row.split = select.value || null;
    drawTotals();
    select.disabled = true;
    try {
      const answer = await post(`/api/library/datasets/${encodeURIComponent(summary.ref)}/splits`, {splits: {[row.recording]: row.split}});
      summary = answer;
      for (const fresh of answer.recordings) Object.assign(rows.find((r) => r.recording === fresh.recording) || {}, fresh);
    } catch (error) {
      row.split = previous;
      select.value = previous || NOT_INCLUDED;
      page.ctx.toast(`Could not change ${row.recording}: ${error.message}`, "error");
    }
    select.disabled = false;
    drawTotals();
  };

  const splitCell = (row) => {
    if (!summary.writable) return el("span", {class: `badge tr-split-${row.split || "none"}`}, SPLITS.find(([key]) => key === row.split)?.[1] || "Not included");
    const select = el("select", {class: `tr-split-select`, "aria-label": `Split of ${row.recording}`, onchange: () => choose(row, select)},
      el("option", {value: NOT_INCLUDED, selected: !row.split}, "Not included"),
      SPLITS.map(([key, label]) => el("option", {value: key, selected: row.split === key}, label)));
    return select;
  };

  const table = el("table", {class: "data tr-recordings"},
    el("thead", {}, el("tr", {}, el("th", {}, "Recording"), el("th", {class: "num"}, "Labels"), el("th", {}, "Status"), el("th", {}, "Split"))),
    el("tbody", {}, rows.map((row) => el("tr", {dataset: {recording: row.recording}},
      el("td", {}, el("button", {class: "link", title: "Browse these labels in Labeling",
        onclick: () => page.ctx.navigate("labeling", `browse/${encodeURIComponent(summary.ref)}/${encodeURIComponent(row.recording)}`)}, row.recording)),
      el("td", {class: "num"}, row.labels),
      el("td", {class: "note"}, statusText(row.statuses)),
      el("td", {}, splitCell(row))))));

  drawTotals();
  container.append(
    el("div", {class: "row tr-dataset-head"},
      el("div", {class: "stack"}, el("strong", {}, summary.name),
        el("span", {class: "mp-sub"}, `${summary.ref}${summary.description ? ` · ${summary.description}` : ""}`
          + (summary.writable ? "" : " · lab datasets are read-only: make a New dataset starting from its splits to change them"))),
      freeze),
    totalsBar,
    rows.length ? table : el("p", {class: "empty"}, "No labels for this setup yet. Label frames in Labeling; their recordings appear here, not included."),
    el("p", {class: "note tr-benchmarks"}, benchmarks.length
      ? `Benchmarks: ${benchmarks.map((b) => `${b.ref} (${b.labels} labels, ${b.recordings.length} recordings)`).join(", ")}.`
      : "No benchmark yet."));
}
