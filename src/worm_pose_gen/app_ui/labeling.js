// The Labeling page (docs/APP_SIMPLIFICATION.md, section 3): label frames
// one at a time, mask then body then Save & next, into the setup's label
// collection (which recordings train, validate or test is each dataset's
// choice, in the Training page's Datasets tab).
//
//   #labeling                    the queues of the setup, New queue, Browse labels
//   #labeling/new                New queue: recordings, a number of frames and optionally image types → Find frames (a job)
//   #labeling/queue/<id>[/<n>]   a queue's frames (Relabel keyframes from the Workspace, found frames, a manifest),
//                                filtered by recording, status and image type and sorted; <n> is the entry's place in the queue
//   #labeling/browse[/<dataset>[/<recording>]]   the setup's labels, filtered, lowest body fit IoU first;
//                                with a dataset, with their recordings' splits in it
//                                (the Training page's Datasets tab links a dataset's recording here)
//
// The left panel walks the frames; the editor (labeling_editor.js) labels the
// open one. A finished Relabel queue offers Back to workspace, which hands the
// queue to the Workspace page to stitch (#workspace/<workspace>/stitch/<id>).
// The setup is chosen in the header and remembered in this browser.

import {api, el, post, query} from "./api.js";
import {FrameEditor, SHORTCUTS} from "./labeling_editor.js";

const STORE_SETUP = "labeling.setup";
const remembered = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
const remember = (key, value) => { try { value ? localStorage.setItem(key, value) : localStorage.removeItem(key); } catch { /* private mode */ } };
const button = (label, onclick, attributes = {}) => el("button", {type: "button", onclick, ...attributes}, label);
const KIND_LABELS = {relabel: "Relabel", spread: "Found frames", manifest: "Manifest"};
// The image types a Find frames job sorts frames into (frame_search.TYPES).
const TYPES = [
  ["contact", "Self-contact", "The body touches or crosses itself"],
  ["edge", "At the edge", "The body reaches the image border"],
  ["pieces", "In pieces", "The worm is predicted in several pieces: debris, a second animal, a broken mask"],
  ["empty", "No worm", "The model finds no worm"],
  ["clear", "Clear", "One whole body in view, apart from itself"],
];
const TYPE_LABELS = Object.fromEntries(TYPES.map(([type, label]) => [type, label]));
const QUEUE_VIEW = {recording: "", status: "", type: "", sort: "order"};

let ctx, root, left, editor;
const page = {
  setups: [], setup: null, mode: "home",
  queue: null, index: -1, queueView: {...QUEUE_VIEW}, browse: {rows: [], token: 0, filters: {recording: "", split: "", status: "", contact: false}},
  poll: null, visible: false,
};

export async function mount(section, context) {
  ctx = context;
  left = el("aside", {class: "panel lb-left"});
  const center = el("div", {class: "lb-main"});
  const side = el("aside", {class: "panel right lb-right"});
  root = el("div", {class: "lb-page"}, left, center, side);
  section.append(root);
  editor = new FrameEditor(ctx, {center, side, onSaved, onChange: () => { renderLeftMarks(); root.classList.toggle("lb-has-frame", editor.open); }});
  document.addEventListener("keydown", onKey);
  try {
    page.setups = (await api("/api/library/setups")).setups;
  } catch (error) {
    ctx.toast(`Could not read the setups: ${error.message}`, "error");
  }
  const stored = remembered(STORE_SETUP);
  page.setup = page.setups.find((s) => s.ref === stored)?.ref || page.setups[0]?.ref || null;
}

export async function show(params = "") {
  page.visible = true;
  const [mode, id, index] = params.split("/").map(decodeURIComponent);
  if (mode === "queue" && id) {
    if (page.mode === "queue" && page.queue?.id === id && index === undefined) { renderHeader(); return; }
    await openQueue(id, index === undefined ? null : Number(index));
  } else if (mode === "browse") {
    await openBrowse(id || null, index || "");
  } else if (mode === "new") {
    if (!leaveFrame()) return;
    page.mode = "new";
    renderNew();
  } else {
    if (!leaveFrame()) return;
    page.mode = "home";
    await renderHome();
  }
  renderHeader();
}

export function hide() {
  if (!leaveFrame()) return false;
  page.visible = false;
  stopPolling();
  return true;
}

// Whether the open frame may be left: no unsaved changes, or the user lets them go.
function leaveFrame() {
  if (!editor.dirty) { editor.stopPlay(); return true; }
  if (!window.confirm("This frame has unsaved changes. Leave without saving?")) return false;
  editor.clear();
  return true;
}

// ------------------------------------------------------------------ header

async function setSetup(ref) {
  if (ref === page.setup) return;
  if (!leaveFrame()) { renderHeader(); return; }
  page.setup = ref;
  remember(STORE_SETUP, ref);
  editor.clear();
  ctx.navigate("labeling");
}

function renderHeader() {
  if (!page.visible) return;
  const context = [];
  if (page.setups.length > 1) {
    const select = el("select", {class: "lb-header-select", "aria-label": "Setup", onchange: (e) => setSetup(e.target.value)},
      page.setups.map((s) => el("option", {value: s.ref, selected: s.ref === page.setup}, s.name)));
    context.push(el("label", {class: "inline"}, "Setup", select));
  } else if (page.setups.length === 1) {
    context.push(el("span", {}, page.setups[0].name));
  }
  if (page.setup) context.push(el("span", {class: "lb-saving", title: "Each dataset of the setup chooses which recordings train, validate or test"},
    "Saving to the setup's labels"));
  ctx.setHeader("labeling", {context, actions: [button("?", showHelp, {title: "Keyboard shortcuts (?)", "aria-label": "Keyboard shortcuts"})]});
}

function showHelp() {
  const rows = SHORTCUTS.map(([key, what]) => el("tr", {}, el("td", {}, el("kbd", {}, key)), el("td", {}, what)));
  const dialog = el("dialog", {class: "lb-help"}, el("h2", {}, "Labeling shortcuts"),
    el("p", {class: "note"}, "Paint the mask, then settle the body (use the proposal, flip it, or trace it), then Save & next."),
    el("table", {class: "data"}, el("tbody", {}, rows)),
    el("div", {class: "row"}, button("Close", () => dialog.close())));
  dialog.addEventListener("close", () => dialog.remove());
  document.body.append(dialog);
  dialog.showModal();
}

// ------------------------------------------------------------------ home: the queues

async function renderHome() {
  editor.clear();
  const body = el("div", {class: "stack"});
  left.replaceChildren(
    el("section", {class: "section"}, el("h3", {}, "Queues"), body),
    el("section", {class: "section"}, el("div", {class: "stack"},
      button("New queue…", () => ctx.navigate("labeling", "new"), {class: "primary", disabled: !page.setup}),
      button("Browse labels", () => ctx.navigate("labeling", "browse"), {disabled: !page.setup}),
    )),
  );
  if (!page.setup) { body.append(el("p", {class: "note"}, "There is no setup yet. Add one with its recordings on the Workspace page.")); return; }
  let queues = [];
  try {
    queues = (await api(`/api/queues${query({setup: page.setup})}`)).queues;
  } catch (error) {
    body.append(el("p", {class: "note error"}, error.message));
    return;
  }
  if (!queues.length) body.append(el("p", {class: "note"}, "No queues yet. Relabel a stretch in a workspace, or make a new queue."));
  const list = el("div", {class: "list lb-queues"});
  for (const queue of queues) list.append(queueRow(queue));
  body.append(list);
  const finding = queues.filter((q) => q.state === "finding");
  if (finding.length) pollJobs(finding, () => page.mode === "home" && renderHome());
}

function queueRow(queue) {
  const {saved, total} = queue.progress;
  const state = queue.state === "finding" ? el("span", {class: "note", dataset: {job: queue.job}}, "finding frames…")
    : queue.state === "failed" ? el("span", {class: "note error"}, queue.error || "failed")
    : el("span", {class: "note"}, queue.complete ? "done" : `${saved} of ${total} saved`);
  const meter = queue.state === "ready" ? el("progress", {max: Math.max(total, 1), value: saved}) : null;
  return el("div", {class: "item lb-queue", dataset: {queue: queue.id}, onclick: () => queue.state === "ready" && ctx.navigate("labeling", `queue/${queue.id}`)},
    el("div", {class: "stack lb-grow"},
      el("div", {class: "row lb-tight"}, el("span", {class: "lb-name"}, queue.name), el("span", {class: "badge"}, KIND_LABELS[queue.kind] || queue.kind)),
      meter, state),
    button("×", (event) => { event.stopPropagation(); removeQueue(queue); }, {class: "lb-remove", title: "Remove this queue (saved labels stay)", "aria-label": "Remove queue"}));
}

async function removeQueue(queue) {
  if (!window.confirm(`Remove the queue "${queue.name}"? Labels saved from it stay in your dataset.`)) return;
  try { await api(`/api/queues/${queue.id}`, {method: "DELETE"}); } catch (error) { ctx.toast(error.message, "error"); }
  renderHome();
}

function pollJobs(queues, done) {
  stopPolling();
  page.poll = setInterval(async () => {
    for (const queue of queues) {
      try {
        const job = await api(`/api/jobs/${queue.job}`);
        const node = left.querySelector(`[data-job="${queue.job}"]`);
        if (node) node.textContent = `finding frames… ${Math.round(100 * job.progress)}%${job.message ? ` · ${job.message}` : ""}`;
        if (["done", "failed", "cancelled"].includes(job.state)) { stopPolling(); done(job, queue); return; }
      } catch { /* the next tick retries */ }
    }
  }, 1000);
}

function stopPolling() { clearInterval(page.poll); page.poll = null; }

// ------------------------------------------------------------------ new queue

async function renderNew() {
  editor.clear();
  const list = el("div", {class: "stack lb-recordings"}, el("p", {class: "note"}, "Loading the recordings…"));
  const frames = el("input", {type: "number", min: 1, max: 500, value: 20, "aria-label": "Frames to find"});
  const status = el("div", {class: "stack"});
  const setup = page.setups.find((s) => s.ref === page.setup);
  const types = el("fieldset", {class: "lb-types", disabled: !setup?.defaults?.mask},
    el("legend", {}, "Image types"),
    ...TYPES.map(([type, label, title]) => el("label", {class: "inline lb-check", title}, el("input", {type: "checkbox", value: type}), label)),
    el("p", {class: "note"}, "Only frames of the checked types; none checked finds any frame."));
  const find = button("Find frames", () => start(), {class: "primary"});
  left.replaceChildren(
    button("← Queues", () => ctx.navigate("labeling"), {class: "link"}),
    el("section", {class: "section"}, el("h3", {}, "New queue"),
      el("p", {class: "note"}, "Pick recordings and how many frames to label. The model looks over each recording and picks frames spread over it, favouring those it is least sure of."),
      list, el("label", {}, "Frames", frames), types, el("div", {class: "row"}, find), status),
  );
  let recordings = [];
  try {
    recordings = (await api(`/api/library/setups/${encodeURIComponent(page.setup)}/recordings`)).recordings;
  } catch (error) {
    list.replaceChildren(el("p", {class: "note error"}, error.message));
    return;
  }
  const readable = recordings.filter((r) => r.readable !== false);
  list.replaceChildren(...(readable.length ? readable.map((r) => el("label", {class: "inline lb-check"},
    el("input", {type: "checkbox", value: r.path}), el("span", {class: "lb-grow"}, r.id), el("span", {class: "note"}, r.frames ? `${r.frames} frames` : "")))
    : [el("p", {class: "note"}, "No readable recordings in this setup.")]));
  if (!setup?.defaults?.mask) status.append(el("p", {class: "note warn"}, "This setup has no model yet: the frames will simply be spread out, of any type."));

  async function start() {
    const chosen = [...list.querySelectorAll("input:checked")].map((input) => input.value);
    const kinds = [...types.querySelectorAll("input:checked")].map((input) => input.value);
    if (!chosen.length) { ctx.toast("Choose at least one recording.", "error"); return; }
    find.disabled = true;
    try {
      const queue = await post("/api/queues", {kind: "spread", setup: page.setup, recordings: chosen, frames: Number(frames.value), types: kinds});
      const meter = el("progress", {max: 1, value: 0}), note = el("span", {class: "note", dataset: {job: queue.job}}, "Starting…");
      status.replaceChildren(meter, note);
      pollJobs([queue], (job) => {
        if (job.state === "done") ctx.navigate("labeling", `queue/${queue.id}`);
        else { status.replaceChildren(el("p", {class: "note error"}, job.error || `The search was ${job.state}.`)); find.disabled = false; }
      });
      const tick = setInterval(async () => {
        if (!page.poll) { clearInterval(tick); return; }
        try { const job = await api(`/api/jobs/${queue.job}`); meter.value = job.progress; } catch { /* retry */ }
      }, 1000);
    } catch (error) {
      status.replaceChildren(el("p", {class: "note error"}, error.message));
      find.disabled = false;
    }
  }
}

// ------------------------------------------------------------------ a queue

async function openQueue(id, index) {
  if (!leaveFrame()) return;
  let queue;
  try {
    queue = await api(`/api/queues/${id}`);
  } catch (error) {
    ctx.toast(`Could not open the queue: ${error.message}`, "error");
    ctx.navigate("labeling");
    return;
  }
  if (queue.setup !== page.setup) { page.setup = queue.setup; remember(STORE_SETUP, queue.setup); }
  if (page.queue?.id !== queue.id) page.queueView = {...QUEUE_VIEW};
  page.mode = "queue";
  page.queue = queue;
  renderQueue();
  const start = index ?? queue.first_unsaved ?? 0;
  if (queue.entries.length) await go(start, {force: true});
}

// The queue's entries (their places in the queue) the view's filters keep, in its order.
function visibleEntries() {
  const list = page.queue.entries, v = page.queueView;
  const kept = list.map((_, k) => k).filter((k) => {
    const entry = list[k];
    return (!v.recording || entry.recording === v.recording) && (!v.status || (v.status === "saved") === Boolean(entry.saved))
      && (!v.type || (entry.types || []).includes(v.type));
  });
  if (v.sort === "uncertainty") kept.sort((a, b) => (list[b].uncertainty ?? -1) - (list[a].uncertainty ?? -1) || a - b);
  else if (v.sort === "frame") kept.sort((a, b) => list[a].recording.localeCompare(list[b].recording) || list[a].frame - list[b].frame);
  return kept;
}

function queueControls() {
  const list = page.queue.entries, v = page.queueView;
  const select = (name, options, label) => el("select", {"aria-label": label, onchange: (e) => { v[name] = e.target.value; renderQueue(); }},
    options.map(([value, text]) => el("option", {value, selected: v[name] === value}, text)));
  const recordings = [...new Set(list.map((entry) => entry.recording))];
  const types = TYPES.filter(([type]) => list.some((entry) => (entry.types || []).includes(type)));
  const sorts = [["order", "Queue order"], ["frame", "Recording, frame"]];
  if (list.some((entry) => entry.uncertainty != null)) sorts.splice(1, 0, ["uncertainty", "Least sure first"]);
  return el("div", {class: "stack lb-filters"},
    recordings.length > 1 ? select("recording", [["", "All recordings"], ...recordings.map((r) => [r, r])], "Recording") : null,
    el("div", {class: "row lb-tight"},
      select("status", [["", "Any status"], ["todo", "To do"], ["saved", "Saved"]], "Status"),
      types.length ? select("type", [["", "Any type"], ...types.map(([type, label]) => [type, label])], "Image type") : null),
    select("sort", sorts, "Sort"));
}

function renderQueue() {
  const queue = page.queue, {saved, total} = queue.progress;
  const visible = visibleEntries();
  const items = visible.map((k) => {
    const entry = queue.entries[k];
    const types = (entry.types || []).map((type) => TYPE_LABELS[type] || type).join(", ");
    return el("div", {class: "item lb-entry", "aria-current": String(k === page.index), dataset: {index: k}, onclick: () => go(k)},
      el("span", {class: `lb-dot ${entry.saved ? "done" : ""}`, "aria-label": entry.saved ? "saved" : "to do"}),
      el("div", {class: "stack lb-grow"},
        el("span", {}, `${entry.recording} · ${entry.frame}`),
        types ? el("span", {class: "note lb-sub"}, types) : null),
      entry.uncertainty != null ? el("span", {class: "note", title: "How unsure the model was (entropy per worm pixel)"}, entry.uncertainty.toFixed(2)) : null);
  });
  const back = queue.kind === "relabel" && queue.complete
    ? el("div", {class: "lb-done"}, el("p", {class: "note"}, "Every keyframe is labeled. The workspace can now stitch the stretch between them."),
      button("Back to workspace", () => backToWorkspace(), {class: "primary"}))
    : queue.complete ? el("p", {class: "note ok lb-done"}, "Every frame of this queue is saved.") : null;
  left.replaceChildren(
    button("← Queues", () => ctx.navigate("labeling"), {class: "link"}),
    el("section", {class: "section"},
      el("h3", {}, KIND_LABELS[queue.kind] || "Queue"),
      el("div", {class: "lb-name"}, queue.name),
      el("progress", {max: Math.max(total, 1), value: saved}),
      el("div", {class: "note"}, `${saved} of ${total} saved`),
      back),
    el("section", {class: "section"}, queueControls(),
      el("div", {class: "note lb-count"}, visible.length === total ? `${total} frame${total === 1 ? "" : "s"}` : `${visible.length} of ${total} frames shown`)),
    el("section", {class: "section lb-fill"}, el("div", {class: "list lb-entries"}, items.length ? items : el("p", {class: "note"}, "No frames match these filters."))),
    el("div", {class: "row lb-nav"}, button("◀ Prev (Shift+←)", () => step(-1), {title: "Previous frame (Shift+←)"}), button("Next (Shift+→) ▶", () => step(1), {title: "Next frame (Shift+→)"})),
  );
  left.querySelector('[aria-current="true"]')?.scrollIntoView({block: "nearest"});
}

function backToWorkspace() {
  if (!leaveFrame()) return;
  ctx.navigate("workspace", `${encodeURIComponent(page.queue.workspace)}/stitch/${page.queue.id}`);
}

// The frames the left panel walks: a queue's entries or the browsed labels.
function entries() {
  if (page.mode === "queue") return page.queue.entries;
  if (page.mode === "browse") return page.browse.rows.map((row) => ({recording: row.recording, frame: row.frame, path: row.source_path}));
  return [];
}

async function go(index, {force = false} = {}) {
  const list = entries();
  if (index < 0 || index >= list.length) return false;
  if (!force && index === page.index) return true;
  if (!leaveFrame()) return false;
  page.index = index;
  if (page.mode === "queue") {
    const hash = `#labeling/queue/${page.queue.id}/${index}`;
    if (location.hash !== hash) history.replaceState(null, "", hash);
  }
  renderLeftMarks();
  const request = {setup: page.setup, entry: list[index]};
  if (page.mode === "queue") request.queue = page.queue.id;
  try {
    await editor.load(request);
  } catch (error) {
    ctx.toast(`Could not open ${list[index].recording} frame ${list[index].frame}: ${error.message}`, "error");
    editor.clear();
  }
  renderHeader();
  return true;
}

// Prev/Next walk the frames in the order the left panel shows them (a queue's filtered and sorted view).
function step(direction) {
  if (page.mode !== "queue") return go(page.index + direction);
  const visible = visibleEntries(), at = visible.indexOf(page.index);
  const target = at < 0 ? visible[direction > 0 ? 0 : visible.length - 1] : visible[at + direction];
  return target === undefined ? false : go(target);
}

// Mark the current entry in the list without rebuilding it.
function renderLeftMarks() {
  for (const node of left.querySelectorAll(".lb-entry")) node.setAttribute("aria-current", String(Number(node.dataset.index) === page.index));
  left.querySelector('.lb-entry[aria-current="true"]')?.scrollIntoView({block: "nearest"});
}

async function onSaved(answer, {next}) {
  if (page.mode === "queue") {
    // The view as it was before this save, so a "To do" filter that now hides this frame still continues after it.
    const order = visibleEntries(), at = order.indexOf(page.index);
    page.queue = await api(`/api/queues/${page.queue.id}`);
    renderQueue();
    if (!next) return;
    // The next unsaved frame of the view after this one, wrapping round to its start.
    const target = [...order.slice(at + 1), ...order.slice(0, at + 1)].find((k) => !page.queue.entries[k].saved);
    if (target !== undefined) go(target);
    else if (!page.queue.complete) ctx.toast("Every frame these filters show is saved; change them to see the rest.", "ok");
    else ctx.toast(page.queue.kind === "relabel" ? "Every keyframe is labeled: go back to the workspace to stitch." : "This queue is done.", "ok");
  } else if (page.mode === "browse") {
    const row = page.browse.rows[page.index];
    if (row) Object.assign(row, answer.label, {fit_iou: null, targets: "missing"});
    renderBrowseList();
    if (next) step(1);
  }
}

// ------------------------------------------------------------------ browse labels

// Browse the setup's labels, optionally of one recording; with `datasetRef` (a dataset of the setup) each shows its
// recording's split there and the split filter applies.
async function openBrowse(datasetRef = null, recording = "") {
  if (!leaveFrame()) return;
  const f = page.browse.filters;
  if (datasetRef) {
    try {
      const summary = await api(`/api/library/datasets/${encodeURIComponent(datasetRef)}`);
      if (summary.setup !== page.setup) { page.setup = summary.setup; remember(STORE_SETUP, summary.setup); }
    } catch (error) {
      ctx.toast(`Could not open ${datasetRef}: ${error.message}`, "error");
      datasetRef = null;
    }
    Object.assign(f, {recording, split: "", status: "", contact: false});
  }
  if (!datasetRef) f.split = "";
  page.mode = "browse";
  page.index = -1;
  page.browse.dataset = datasetRef;
  editor.clear();
  const select = (name, options, label) => el("select", {"aria-label": label, onchange: (e) => { f[name] = e.target.value; loadBrowse(); }},
    options.map(([value, text]) => el("option", {value, selected: f[name] === value}, text)));
  let recordings = [];
  if (page.setup) {
    try { recordings = (await api(`/api/library/setups/${encodeURIComponent(page.setup)}/collection`)).recordings.map((r) => r.recording); } catch { recordings = []; }
  }
  const contact = el("input", {type: "checkbox", checked: f.contact, onchange: (e) => { f.contact = e.target.checked; loadBrowse(); }});
  page.browse.list = el("div", {class: "list lb-entries"});
  page.browse.count = el("div", {class: "note"});
  left.replaceChildren(
    button("← Queues", () => ctx.navigate("labeling"), {class: "link"}),
    el("section", {class: "section"}, el("h3", {}, "Browse labels"),
      recordings.length ? el("p", {class: "note lb-dataset"}, datasetRef ? `The setup's labels, with their splits in ${datasetRef}` : "The setup's labels")
        : el("p", {class: "note"}, "There are no labels for this setup yet."),
      select("recording", [["", "All recordings"], ...recordings.map((r) => [r, r])], "Recording"),
      el("div", {class: "row lb-tight"},
        datasetRef ? select("split", [["", "All splits"], ["train", "Train"], ["val", "Validation"], ["test", "Test"], ["none", "Not included"]], "Split") : null,
        select("status", [["", "Any status"], ["complete", "Complete"], ["mask_only", "Mask only"], ["auto", "Body unconfirmed"]], "Status")),
      el("label", {class: "inline", title: "Bodies that touch or cross themselves: the hard frames"}, contact, "Self-contact only"),
      page.browse.count),
    el("section", {class: "section lb-fill"}, el("div", {class: "note lb-sort"}, "Lowest body fit IoU first"), page.browse.list),
    el("div", {class: "row lb-nav"}, button("◀ Prev (Shift+←)", () => step(-1), {title: "Previous frame (Shift+←)"}), button("Next (Shift+→) ▶", () => step(1), {title: "Next frame (Shift+→)"})),
  );
  await loadBrowse();
}

async function loadBrowse() {
  const f = page.browse.filters, token = ++page.browse.token;
  if (!page.setup) { page.browse.rows = []; renderBrowseList(); return; }
  let rows = [];
  try {
    rows = (await api(`/api/library/setups/${encodeURIComponent(page.setup)}/labels${query({
      dataset: page.browse.dataset, recording: f.recording, split: f.split, status: f.status, contact: f.contact ? "yes" : "", sort: "fit_iou",
    })}`)).labels;
  } catch (error) {
    ctx.toast(error.message, "error");
  }
  if (token !== page.browse.token) return;  // a newer filter's answer is coming
  page.browse.rows = rows;
  page.index = -1;
  renderBrowseList();
}

const STATUS_TEXT = {complete: "complete", mask_only: "mask only", auto: "body unconfirmed"};

function renderBrowseList() {
  const rows = page.browse.rows;
  page.browse.count.textContent = `${rows.length} label${rows.length === 1 ? "" : "s"}`;
  page.browse.list.replaceChildren(...rows.map((row, k) => el("div", {class: "item lb-entry lb-label", "aria-current": String(k === page.index), dataset: {index: k}, onclick: () => go(k)},
    el("div", {class: "stack lb-grow"},
      el("span", {}, `frame ${row.frame}`),
      el("span", {class: "note lb-sub", title: row.recording}, `${STATUS_TEXT[row.status] || row.status} · ${row.recording}`)),
    row.split ? el("span", {class: "badge"}, row.split) : null,
    el("span", {class: "lb-iou", title: row.targets === "built" ? "Body fit IoU: how well the body fitted to the mask covers it" : "Body targets not built yet"},
      row.fit_iou == null ? "—" : row.fit_iou.toFixed(3)))));
}

// ------------------------------------------------------------------ keys

function onKey(event) {
  if (!page.visible || event.defaultPrevented || document.querySelector("dialog[open]")) return;
  const target = event.target;
  // Typing goes to text fields; sliders and checkboxes keep only the keys they use themselves.
  if (target instanceof HTMLElement && (target.isContentEditable || ["SELECT", "TEXTAREA"].includes(target.tagName))) return;
  if (target instanceof HTMLInputElement) {
    if (!["range", "checkbox"].includes(target.type)) return;
    if (target.type === "range" && event.key.startsWith("Arrow")) return;
  }
  if (event.key === "?") { showHelp(); event.preventDefault(); return; }
  if (event.shiftKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
    step(event.key === "ArrowLeft" ? -1 : 1);
    event.preventDefault();
    return;
  }
  if (target instanceof HTMLButtonElement && (event.key === "Enter" || event.key === " ")) return;
  if (editor.handleKey(event)) event.preventDefault();
}
