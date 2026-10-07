// The Recordings screen (docs/APP_SIMPLIFICATION.md, section 2): the chosen
// setup's recordings in one table with their status, and one action per
// row, Analyse or Open. Analyse uses the setup's default models (Change
// picks others for this session) on the whole recording, where Run on says.
// Add recording registers a file to a setup through the library. The table
// rescans whenever the screen opens and polls while an analysis runs.

import {api, el, post, query} from "./api.js";
import {RunOn, loadCompute, modelCards, modelLabel, openAnalyseDialog} from "./workspace_analyse.js";

const POLL_MS = 3000;
const SETUP_KEY = "workspace.setup";

function remembered(key) { try { return localStorage.getItem(key) || ""; } catch { return ""; } }
function remember(key, value) { try { localStorage.setItem(key, value); } catch { /* storage blocked */ } }

export function formatDuration(frames, fps) {
  if (!frames || !fps) return "";
  const seconds = Math.round(frames / fps), m = Math.floor(seconds / 60), s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function formatWhen(iso) {
  if (!iso) return "—";
  const when = new Date(iso), days = (Date.now() - when.getTime()) / 864e5;
  if (days < 1 && when.getDate() === new Date().getDate()) return when.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
  return when.toLocaleDateString([], {month: "short", day: "numeric", year: days > 300 ? "numeric" : undefined});
}

// The status cell of a recording (and the Workspace header's progress): [text, kind, progress | null].
export function describeStatus(status) {
  if (!status) return ["Not analysed", "", null];
  const job = status.analysis;
  switch (status.state) {
    case "queued": return [job?.slurm_state === "PENDING" ? "Waiting for SLURM" : "Queued", "info", null];
    case "analysing": return [`Analysing · ${stageOf(job)}`, "info", job?.progress ?? 0];
    case "failed": return ["Analysis failed", "error", null];
    case "not_analysed": return ["Not analysed", "", null];
    case "analysed": return ["Analysed · open to review", "ok", null];
    case "issues": return [`${status.issues.unreviewed} issue${status.issues.unreviewed === 1 ? "" : "s"} to review`, "warn", null];
    case "reviewed": return ["Reviewed", "ok", null];
    case "exported": return ["Exported", "ok", null];
    default: return [status.state, "", null];
  }
}

const STAGE_WORDS = {segment: "masks", prior: "body size", fit: "fitting", ambiguity: "checking", propagate: "tracking", track: "lengths", fixed_body: "fixed body"};
export function stageOf(job) {
  const stage = String(job?.message || "").split(":")[0];
  return `${STAGE_WORDS[stage] || "starting"} ${Math.round((job?.progress || 0) * 100)}%`;
}

export class RecordingsScreen {
  constructor(section, ctx, {onOpen}) {
    this.ctx = ctx;
    this.onOpen = onOpen;
    this.data = null;
    this.models = null; // the models chosen for this session per setup: {setup, mask, body}
    this.filter = "";
    this.timer = null;
    this.node = el("div", {class: "ws-home", hidden: true});
    section.append(this.node);
  }

  async show() {
    this.node.hidden = false;
    this.ctx.setHeader("workspace", {context: [el("span", {class: "ws-title"}, "Recordings")]});
    if (!this.runOn) {
      try { this.runOn = new RunOn(await loadCompute()); } catch (error) { this.ctx.toast(`Could not check where jobs can run: ${error.message}`, "error"); }
    }
    await this.refresh();
  }

  hide() {
    this.node.hidden = true;
    clearTimeout(this.timer);
  }

  async refresh() {
    clearTimeout(this.timer);
    const setup = this.data?.setup?.ref || remembered(SETUP_KEY);
    try {
      this.data = await api(`/api/home${query({setup})}`);
    } catch (error) {
      if (setup && error.status === 404) { remember(SETUP_KEY, ""); this.data = await api("/api/home"); }
      else { this.renderError(error); return; }
    }
    if (this.data.setup && this.models?.setup !== this.data.setup.ref) await this.defaultModels();
    this.render();
    const busy = this.data.recordings.some((r) => ["queued", "analysing"].includes(r.status?.state));
    if (busy && !this.node.hidden) this.timer = setTimeout(() => this.refresh(), POLL_MS);
  }

  async defaultModels() {
    const setup = this.data.setup, defaults = setup.defaults || {};
    let known = new Map();
    try { known = await modelCards(setup.ref); } catch { /* names fall back to refs */ }
    const model = (ref) => (ref ? {ref, name: known.get(ref)?.name || ref} : null);
    this.models = {setup: setup.ref, mask: model(defaults.mask), body: model(defaults.body)};
  }

  renderError(error) {
    this.node.replaceChildren(el("div", {class: "empty"}, el("p", {}, "Could not list the recordings."), el("p", {class: "note error"}, error.message),
      el("button", {onclick: () => this.refresh()}, "Try again")));
  }

  render() {
    const {setups, setup, recordings} = this.data;
    if (!setup) {
      const libraries = this.ctx.config?.libraries || {};
      this.node.replaceChildren(el("div", {class: "empty ws-empty"},
        el("h2", {}, "No microscope setups yet"),
        el("p", {}, "A setup says how to read a microscope's videos and which models to use on them. Setups come from the lab library or your own."),
        el("p", {class: "note"}, `Lab library: ${libraries.lab || "none on this machine"}${libraries.lab && !libraries.lab_available ? " (not reachable)" : ""} · yours: ${libraries.personal || "?"}`)));
      return;
    }
    const setupControl = setups.length > 1
      ? el("label", {class: "inline"}, "Setup", el("select", {onchange: (event) => { remember(SETUP_KEY, event.target.value); this.data.setup = {ref: event.target.value}; this.refresh(); }},
        setups.map((s) => el("option", {value: s.ref, selected: s.ref === setup.ref}, s.name))))
      : el("span", {class: "ws-setup-name", title: setup.ref}, setup.name);
    const facts = [setup.pixel_size_um ? `${setup.pixel_size_um} µm/px` : null, setup.fps ? `${setup.fps} fps` : null].filter(Boolean).join(" · ");
    const toolbar = el("div", {class: "ws-home-bar"},
      setupControl, facts ? el("span", {class: "note"}, facts) : null,
      el("span", {class: "ws-sep"}),
      el("span", {class: "ws-analyse-with"}, "Analyse with ", el("strong", {}, modelLabel(this.models)), " ",
        el("button", {class: "link", onclick: () => this.changeModels()}, "Change")),
      this.runOn?.control() ?? null,
      el("span", {class: "ws-grow"}),
      el("input", {type: "search", class: "ws-filter", placeholder: "Filter recordings", value: this.filter, oninput: (event) => { this.filter = event.target.value; this.renderRows(); }}),
      el("button", {onclick: () => this.addRecording()}, "Add recording…"),
    );
    if (this.runOn && !this.runOn.canRun) toolbar.append(el("p", {class: "note error ws-home-reason"}, `Analysis is unavailable: ${this.runOn.reason}`));
    this.tbody = el("tbody");
    const table = el("table", {class: "data ws-recordings"},
      el("thead", {}, el("tr", {}, el("th", {class: "ws-thumb-cell"}), el("th", {}, "Recording"), el("th", {class: "num"}, "Length"),
        el("th", {}, "Status"), el("th", {}, "Last opened"), el("th", {class: "ws-action-cell"}))),
      this.tbody);
    this.node.replaceChildren(toolbar, el("div", {class: "ws-table-wrap"}, table));
    this.renderRows();
    if (!recordings.length) this.tbody.append(el("tr", {}, el("td", {colspan: 6, class: "empty"}, "No recordings in this setup yet. Add one with Add recording.")));
  }

  renderRows() {
    if (!this.tbody) return;
    const fps = this.data.setup.fps, needle = this.filter.trim().toLowerCase();
    const rows = this.data.recordings.filter((r) => !needle || r.id.toLowerCase().includes(needle));
    this.tbody.replaceChildren(...rows.map((row) => this.row(row, fps)));
  }

  row(recording, fps) {
    const status = recording.status;
    const [text, kind, progress] = recording.readable ? describeStatus(status) : ["Cannot be read", "error", null];
    const middle = Math.floor((recording.frames || 1) / 2);
    const thumb = recording.readable
      ? el("img", {class: "ws-thumb", loading: "lazy", alt: "", src: `/api/recordings/thumbnail${query({path: recording.path, frame: middle, scale: 0.12})}`})
      : el("span", {class: "ws-thumb"});
    const statusCell = el("td", {class: "ws-status"}, el("span", {class: `badge ${kind}`, title: status?.analysis?.error || recording.error || null}, text));
    if (progress !== null) statusCell.append(el("progress", {max: 1, value: progress}));
    const analysed = status && !["not_analysed", "failed"].includes(status.state);
    const action = analysed
      ? el("button", {class: "primary", onclick: () => this.onOpen(status.name)}, "Open")
      : el("button", {class: "primary", disabled: !recording.readable || !this.runOn?.canRun || !this.models?.mask,
          title: !this.models?.mask ? "This setup has no default mask model; Change to choose one" : this.runOn && !this.runOn.canRun ? this.runOn.reason : null,
          onclick: (event) => this.analyse(recording, event.currentTarget)}, status?.state === "failed" ? "Analyse again" : "Analyse");
    return el("tr", {dataset: {recording: recording.id}},
      el("td", {class: "ws-thumb-cell"}, thumb),
      el("td", {}, el("div", {class: "ws-rec-name"}, recording.id), el("div", {class: "note ws-rec-path", title: recording.path}, recording.path)),
      el("td", {class: "num"}, recording.frames ? `${recording.frames.toLocaleString()} fr` : "—", fps && recording.frames ? el("div", {class: "note"}, formatDuration(recording.frames, fps)) : null),
      statusCell,
      el("td", {class: "note"}, formatWhen(status?.last_opened)),
      el("td", {class: "ws-action-cell"}, action),
    );
  }

  async analyse(recording, button, extra = {}) {
    button.disabled = true;
    try {
      const answer = await post("/api/analyse", {path: recording.path, models: {mask: this.models.mask?.ref ?? null, body: this.models.body?.ref ?? null}, ...this.runOn.payload(), ...this.devOptions, ...extra});
      this.ctx.toast(`Analysing ${recording.id}`, "ok");
      if (answer) await this.refresh();
    } catch (error) {
      this.ctx.toast(error.message, "error");
      button.disabled = false;
    }
  }

  async changeModels() {
    const chosen = await openAnalyseDialog(this.ctx, {
      title: "Analyse with", setup: this.data.setup.ref, models: this.models, submitLabel: "Use these models", dev: this.ctx.dev, newWorkspace: null,
    });
    if (!chosen) return;
    this.models = {setup: this.data.setup.ref, mask: chosen.mask, body: chosen.body};
    this.devOptions = chosen.stages ? {stages: chosen.stages, ...(chosen.gpu !== undefined ? {gpu: chosen.gpu} : {})} : null;
    this.render();
  }

  addRecording() {
    openAddRecording(this.ctx, this.data.setups, this.data.setup.ref).then((added) => {
      if (!added) return;
      if (added.setup !== this.data.setup.ref) { remember(SETUP_KEY, added.setup); this.data.setup = {ref: added.setup}; }
      this.ctx.toast(`Added ${added.id}`, "ok");
      this.refresh();
    });
  }
}

// Add recording: a path box with a browser of directories and HDF5 files, and the setup it belongs to.
function openAddRecording(ctx, setups, current) {
  return new Promise((resolve) => {
    let result = null, listing = null;
    const dialog = el("dialog", {class: "ws-dialog ws-add"});
    const path = el("input", {type: "text", placeholder: "/path/to/recording.h5", "aria-label": "Recording path"});
    const list = el("div", {class: "list ws-browse"});
    const crumbs = el("div", {class: "note ws-browse-at"});
    const error = el("p", {class: "note error", hidden: true});
    const setup = el("select", {"aria-label": "Setup"}, setups.map((s) => el("option", {value: s.ref, selected: s.ref === current}, s.name)));
    const add = el("button", {class: "primary", type: "button", disabled: true}, "Add");
    const browse = async (at) => {
      try {
        listing = await api(`/api/files${query({path: at})}`);
      } catch (failure) { error.textContent = failure.message; error.hidden = false; return; }
      error.hidden = true;
      crumbs.textContent = listing.path;
      const entries = listing.entries.filter((entry) => entry.kind === "dir" || entry.kind === "h5");
      list.replaceChildren(
        ...(listing.parent ? [el("div", {class: "item", onclick: () => browse(listing.parent)}, "↑ ..")] : []),
        ...entries.map((entry) => el("div", {class: "item", onclick: () => {
          if (entry.kind === "dir") browse(entry.path);
          else { path.value = entry.path; add.disabled = false; }
        }}, entry.kind === "dir" ? `▸ ${entry.name}` : entry.name, entry.kind === "h5" && entry.size_bytes ? el("span", {class: "note"}, ` ${(entry.size_bytes / 1e9).toFixed(1)} GB`) : null)),
      );
    };
    path.addEventListener("input", () => { add.disabled = !path.value.trim(); });
    path.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      const value = path.value.trim();
      if (/\.(h5|hdf5)$/i.test(value)) add.click(); else if (value) browse(value);
    });
    add.addEventListener("click", async () => {
      add.disabled = true;
      try {
        result = await post("/api/library/recordings", {path: path.value.trim(), setup: setup.value});
        dialog.close();
      } catch (failure) { error.textContent = failure.message; error.hidden = false; add.disabled = false; }
    });
    dialog.append(
      el("h2", {}, "Add a recording"),
      el("div", {class: "ws-form-row"}, path, el("button", {type: "button", onclick: () => browse(path.value.trim() || undefined)}, "Browse")),
      crumbs, list, error,
      el("div", {class: "ws-form-row"}, el("span", {class: "ws-form-label"}, "Setup"), setup),
      el("div", {class: "ws-dialog-actions"}, el("button", {type: "button", onclick: () => dialog.close()}, "Cancel"), add),
    );
    dialog.addEventListener("close", () => { dialog.remove(); resolve(result); });
    document.body.append(dialog);
    dialog.showModal();
    path.focus();
  });
}
