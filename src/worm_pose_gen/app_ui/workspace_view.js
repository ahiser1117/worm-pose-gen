// One workspace (one recording): the Issues panel and the four fixes on the
// left, the frame in the middle, transport, issue track and kymograph at
// the bottom (docs/APP_SIMPLIFICATION.md, section 2).
//
// The fixes act on a *target*: the selected issue, or a range dragged on the
// timeline (which replaces the issue), or the current frame when there is
// neither. Refit and Relabel's stitch run as jobs whose result is a preview
// drawn on the frame (before dashed, after in pink) to Keep or Discard; Edit
// mask saves the frame's mask and refits around it with a job that keeps
// its own result; Flip, Keep, Undo and Looks OK answer at once. After any
// change the issues, the fixes list, the kymograph and the frame reload.
//
// Frames load light (image and pose) while playing or stepping and in full
// (mask and every layer) once the cursor rests. Server endpoints are under
// /api/workspaces/<name>: status, issues, fixes, kymograph, frame, mask.

import {api, el, post, query} from "./api.js";
import {FrameCanvas, drawMidline, maskCanvas} from "./frame_canvas.js";
import {RunOn, loadCompute, modelCards, modelLabel, openAnalyseDialog} from "./workspace_analyse.js";
import {AP_LUT, KYMOGRAPH_LUT, colorize, decodeGray, drawCurve, drawOutline, loadImage} from "./workspace_draw.js";
import {describeStatus} from "./workspace_home.js";
import {MaskEditor} from "./workspace_mask.js";
import {Timeline} from "./workspace_timeline.js";
import {DevTools} from "./workspace_dev.js";

const STATUS_POLL_MS = 2000, JOB_POLL_MS = 700, SETTLE_MS = 180;
const SPEEDS = [0.5, 1, 2, 4];
const LAYERS = [
  {id: "mask", label: "Mask", key: "1"},
  {id: "midline", label: "Midline", key: "2", title: "Midline with head (square) and tail (circle)"},
  {id: "outline", label: "Outline", key: "3", title: "Body outline"},
  {id: "ap", label: "A-P field", key: "4", title: "Head-to-tail field of the body model"},
];
const STATE_WORDS = {unreviewed: "to review", reviewed: "looks OK", fixed: "fixed"};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const range = ([a, b]) => (a === b ? `${a}` : `${a}–${b}`);

export class WorkspaceView {
  constructor(section, ctx, {onHome}) {
    this.ctx = ctx;
    this.onHome = onHome;
    this.name = null;
    this.layers = {mask: true, midline: true, outline: false, ap: false};
    this.speed = 1;
    this.loop = false;
    this._build(section);
    this.mask = null;
    this.dev = ctx.dev ? new DevTools(this) : null;
    document.addEventListener("keydown", (event) => this._key(event));
  }

  // ------------------------------------------------------------------ layout

  _build(section) {
    this.issuesNode = el("div", {class: "ws-issues"});
    this.fixNode = el("div", {class: "ws-fix"});
    this.fixesNode = el("div", {class: "ws-fixes"});
    this.left = el("aside", {class: "ws-left"}, this.issuesNode, this.fixNode, this.fixesNode);

    this.layerButtons = LAYERS.map((layer) => el("button", {
      type: "button", class: "ws-layer", dataset: {layer: layer.id}, title: `${layer.title || layer.label} (${layer.key})`,
      onclick: () => this.toggleLayer(layer.id),
    }, layer.label));
    this.toolbar = el("div", {class: "ws-toolbar"}, el("div", {class: "ws-layers", role: "group", "aria-label": "Layers"}, this.layerButtons),
      el("button", {type: "button", class: "ws-fit", title: "Fit the frame to the view (double-click)", onclick: () => this.canvas.fit()}, "Fit"));
    this.canvasHost = el("div", {class: "ws-canvas"});
    this.legend = el("div", {class: "ws-legend", hidden: true});
    this.frameStatus = el("div", {class: "ws-frame-status"});
    this.empty = el("div", {class: "ws-canvas-empty", hidden: true});
    this.center = el("div", {class: "ws-center"}, this.toolbar, this.canvasHost, this.legend, this.empty, this.frameStatus);

    this.frameInput = el("input", {type: "number", class: "ws-frame-input", min: 0, "aria-label": "Frame",
      onkeydown: (event) => { if (event.key === "Enter") { this.seekFrame(Number(this.frameInput.value)); this.frameInput.blur(); } }});
    this.frameTotal = el("span", {class: "note"});
    this.playButton = el("button", {type: "button", class: "ws-play", title: "Play / pause (Space)", onclick: () => this.togglePlay()}, "Play");
    this.speedSelect = el("select", {"aria-label": "Speed", title: "Playback speed", onchange: (event) => { this.speed = Number(event.target.value); }},
      SPEEDS.map((s) => el("option", {value: s, selected: s === 1}, `${s}×`)));
    this.loopButton = el("button", {type: "button", "aria-pressed": "false", title: "Loop the selected issue or range", onclick: () => {
      this.loop = !this.loop; this.loopButton.setAttribute("aria-pressed", String(this.loop));
    }}, "Loop");
    this.selectionNote = el("span", {class: "ws-selection"});
    this.transport = el("div", {class: "ws-transport"},
      el("button", {type: "button", title: "Previous frame (←)", "aria-label": "Previous frame", onclick: () => this.step(-1)}, "◀"),
      this.playButton,
      el("button", {type: "button", title: "Next frame (→)", "aria-label": "Next frame", onclick: () => this.step(1)}, "▶"),
      el("label", {class: "inline"}, "Frame", this.frameInput), this.frameTotal,
      el("label", {class: "inline"}, "Speed", this.speedSelect), this.loopButton, this.selectionNote,
      el("span", {class: "ws-grow"}),
      el("span", {class: "note ws-timeline-hint"}, "Drag to select · wheel to zoom"));
    this.timelineHost = el("div", {class: "ws-timeline"});
    this.bottom = el("div", {class: "ws-bottom"}, this.transport, this.timelineHost);
    this.node = el("div", {class: "ws-main", hidden: true}, this.left, this.center, this.bottom);
    section.append(this.node);

    this.canvas = new FrameCanvas(this.canvasHost);
    this.canvas.onPointer = (event) => {
      if (this.mask?.active && this.mask.pointer(event)) {
        this.canvas.redraw();
        if (event.type === "up") this.renderFix();
      }
    };
    this.canvasHost.addEventListener("pointerleave", () => { if (this.mask?.active) { this.mask.cursor = null; this.canvas.redraw(); } });
    this._layers();
    this.timeline = new Timeline(this.timelineHost, {
      onSeek: (row) => this.seek(row),
      onSelect: (rows) => this.select({first: rows.first, last: rows.last, kind: "range"}),
      onKeyframe: (row) => this.toggleKeyframe(row),
    });
  }

  _layers() {
    const view = this.canvas;
    view.setLayer("mask", (g) => { if (this.maskOverlay && !this.mask?.active) g.drawImage(this.maskOverlay, 0, 0); });
    view.setLayer("ap", (g) => { if (this.apCanvas) g.drawImage(this.apCanvas, 0, 0); });
    view.setLayer("dev", (g, v) => this.dev?.drawLayers(g, v));
    view.setLayer("outline", (g, v) => { const pose = this.current?.pose; if (pose) drawOutline(g, v, pose.centerline_xy, pose.width_profile); });
    view.setLayer("midline", (g, v) => { if (this.current?.pose && !this.previewFrame()) drawMidline(g, v, this.current.pose.centerline_xy); });
    view.setLayer("preview", (g, v) => {
      const entry = this.previewFrame();
      if (!entry) return;
      if (entry.before) drawCurve(g, v, entry.before.centerline_xy, {color: "rgba(255,255,255,0.85)", width: 2, dash: [6, 5]});
      drawCurve(g, v, entry.after.centerline_xy, {color: "#ff7ad9", width: 2.5});
    });
    view.setLayer("draft", (g, v) => this.mask?.draw(g, v));
    for (const layer of LAYERS) view.setLayerVisible(layer.id, this.layers[layer.id]);
  }

  // ------------------------------------------------------------- open/close

  async open(name) {
    if (this.name !== name) {
      this.stopPlay();
      this.reset(name);
      this.node.hidden = false;
      post(`${this.base}/opened`).catch(() => {});
      await this.loadStatus();
      if (!this.status) return;
      this.timeline.setRange(this.rows, this.status.frames[0], this.status.step);
      this.frameTotal.textContent = `of ${this.status.frames[0]}–${this.status.frames[1]}`;
      this.frameInput.min = this.status.frames[0]; this.frameInput.max = this.status.frames[1];
      await this.loadAll();
      if (!this.selectFirstIssue()) this.seek(0);
    } else {
      this.node.hidden = false;
      await this.loadStatus();
      if (this.status) await this.loadAll();
    }
    this.renderHeader();
  }

  reset(name) {
    clearTimeout(this.statusTimer);
    this.name = name;
    this.base = `/api/workspaces/${encodeURIComponent(name)}`;
    this.status = null; this.issues = null; this.issuesError = null; this.fixes = []; this.target = null; this.selectedIssue = null;
    this.mode = "idle"; this.pending = null; this.preview = null; this.keyframes = null;
    this.row = 0; this.current = null; this.maskOverlay = null; this.apCanvas = null; this.apCache = new Map();
    this.generation = 0; this.kymographStamp = null;
    this.mask = new MaskEditor(name);
    this.timeline.set("issues", []); this.timeline.set("kymograph", null); this.timeline.set("selection", null);
    this.timeline.set("preview", null); this.timeline.set("keyframes", null); this.timeline.set("selected", null);
    this.dev?.reset();
  }

  hide() {
    this.stopPlay();
    this.node.hidden = true;
    clearTimeout(this.statusTimer);
  }

  // Whether the page may be left: a mask draft asks first.
  canLeave() {
    if (!this.mask?.dirty) return true;
    if (!window.confirm(`Discard your mask edit on frame ${this.mask.frame}?`)) return false;
    this.stopMaskEdit();
    return true;
  }

  get rows() { return this.status?.frame_count || 0; }
  frameOf(row) { return this.status.frames[0] + row * this.status.step; }
  rowOf(frame) { return Math.max(0, Math.min(this.rows - 1, Math.round((frame - this.status.frames[0]) / this.status.step))); }
  get fps() { return this.status?.setup?.fps || 20; }

  // ------------------------------------------------------------ server state

  async loadStatus() {
    try {
      this.status = await api(`${this.base}/status`);
    } catch (error) {
      this.ctx.toast(`Could not open ${this.name}: ${error.message}`, "error");
      if (error.status === 404) this.onHome();
      return;
    }
    this.renderHeader();
    clearTimeout(this.statusTimer);
    const analysing = ["queued", "analysing"].includes(this.status.state);
    if (analysing) this.statusTimer = setTimeout(() => this.pollStatus(), STATUS_POLL_MS);
    if (analysing || this.issuesError) this.renderIssues();
  }

  async pollStatus() {
    const before = this.status?.state;
    await this.loadStatus();
    if (["queued", "analysing"].includes(before) && !["queued", "analysing"].includes(this.status?.state)) {
      if (this.status.state === "failed") this.ctx.toast(`Analysis failed: ${this.status.analysis?.error?.split("\n").pop() || ""}`, "error");
      else this.ctx.toast("Analysis finished", "ok");
      await this.loadAll();
      if (!this.selectFirstIssue()) this.seek(this.row);
    }
  }

  // Start the review on the first issue to review; whether there was one.
  selectFirstIssue() {
    const first = this.issues?.issues.find((issue) => issue.state === "unreviewed");
    if (first) this.selectIssue(first);
    return !!first;
  }

  async loadAll() {
    this.renderLayers();
    await Promise.all([this.loadIssues(), this.loadFixes(), this.loadKymograph()]);
    if (this.status.analysed && this.mode === "idle") await this.resumePreview();
    this.renderFix();
    this.dev?.load();
  }

  async loadIssues() {
    if (!this.status.analysed || ["queued", "analysing"].includes(this.status.state)) {
      this.issues = null; this.issuesError = null;
      this.timeline.set("issues", []);
      this.renderIssues();
      return;
    }
    try {
      this.issues = await api(`${this.base}/issues`);
      this.issuesError = null;
    } catch (error) {
      // 409: a job holds the workspace (its progress shows instead).
      this.issuesError = error;
      if (error.status === 409) { clearTimeout(this.statusTimer); this.statusTimer = setTimeout(() => this.retryIssues(), STATUS_POLL_MS); }
    }
    this.syncIssues();
  }

  async retryIssues() { await this.loadIssues(); this.renderFix(); }

  syncIssues() {
    const list = this.issues?.issues || [];
    this.timeline.set("issues", list);
    if (this.selectedIssue && !list.some((issue) => issue.id === this.selectedIssue)) {
      // The issue changed shape (a fix nearby): follow the issue now under the old target.
      const replacement = this.target && list.find((issue) => issue.rows[0] <= this.target.last && issue.rows[1] >= this.target.first);
      if (replacement) this.select({first: replacement.rows[0], last: replacement.rows[1], kind: "issue", issue: replacement.id}, {seek: false});
      else this.select(null);
    }
    this.renderIssues();
    this.renderStatusLine();
  }

  async loadFixes() {
    if (!this.status.analysed) { this.fixes = []; this.renderFixes(); return; }
    try { this.fixes = (await api(`${this.base}/fixes`)).fixes; } catch (error) { this.fixes = []; }
    this.renderFixes();
  }

  async loadKymograph() {
    if (!this.status.analysed) { this.timeline.set("kymograph", null); return; }
    try {
      const payload = await api(`${this.base}/kymograph`);
      const gray = await decodeGray(payload.image);
      this.timeline.set("kymograph", colorize(gray, KYMOGRAPH_LUT));
    } catch (error) {
      this.timeline.set("kymograph", null);
    }
  }

  // After an edit: the edit response's frame first, then everything that may have changed.
  async changed(response = null) {
    if (response?.fixes) { this.fixes = response.fixes; this.renderFixes(); }
    await Promise.all([this.loadIssues(), response?.fixes ? null : this.loadFixes(), this.loadKymograph()]);
    this.apCache.clear();
    await this.loadFrame(this.row, "full");
    this.renderFix();
  }

  // ---------------------------------------------------------------- header

  renderHeader() {
    if (!this.status || this.node.hidden) return;
    const status = this.status, [text, kind, progress] = describeStatus(status);
    const progressNode = ["queued", "analysing", "failed"].includes(status.state)
      ? el("span", {class: `ws-progress ${kind}`, title: status.analysis?.error || status.analysis?.message || null},
        progress !== null ? el("progress", {max: 1, value: progress}) : null, el("span", {}, text),
        status.state === "failed" && status.analysis ? el("button", {class: "link", onclick: () => this.retryAnalysis()}, "Retry") : null)
      : null;
    const context = [
      el("button", {class: "ws-back", title: "All recordings", onclick: () => this.leaveTo(() => this.onHome())}, "‹ Recordings"),
      el("strong", {class: "ws-title", title: status.recording}, status.recording_id),
      el("span", {class: "ws-dot"}, "·"),
      el("span", {class: "ws-models", title: [status.models.mask?.ref, status.models.body?.ref].filter(Boolean).join(" + ") || null}, modelLabel(status.models)),
      progressNode ? el("span", {class: "ws-dot"}, "·") : null, progressNode,
    ];
    const busy = ["queued", "analysing"].includes(status.state);
    const actions = [
      this.ctx.dev ? el("button", {class: "dev-only", onclick: () => this.dev.openJobs()}, "Jobs") : null,
      el("button", {disabled: busy, title: busy ? "Wait for the analysis to finish" : "Analyse again, with these or other models", onclick: () => this.reanalyse()}, "Change model"),
      el("button", {class: "primary", disabled: !status.analysed || busy, onclick: () => this.export()}, "Export"),
      el("button", {class: "ws-help-button", title: "Keyboard shortcuts (?)", "aria-label": "Help", onclick: () => this.help()}, "?"),
    ];
    this.ctx.setHeader("workspace", {context: context.filter(Boolean), actions: actions.filter(Boolean)});
  }

  leaveTo(action) { if (this.canLeave()) action(); }

  async retryAnalysis() {
    try {
      await post(`/api/jobs/${this.status.analysis.id}/retry`);
      await this.loadStatus();
      this.renderIssues();
    } catch (error) { this.ctx.toast(error.message, "error"); }
  }

  async reanalyse(first = false) {
    const status = this.status;
    let runOn;
    try { runOn = new RunOn(await loadCompute()); } catch (error) { this.ctx.toast(error.message, "error"); return; }
    const models = {mask: status.models.mask, body: status.models.body};
    if (!models.mask && status.setup) {
      const known = await modelCards(status.setup.ref).catch(() => new Map());
      const ref = (await api(`/api/library/setups/${encodeURIComponent(status.setup.ref)}`).catch(() => ({}))).defaults?.mask;
      if (ref) models.mask = {ref, name: known.get(ref)?.name || ref};
    }
    if (!status.setup) { this.ctx.toast("This recording does not belong to a setup; add it to one on the Recordings screen.", "error"); return; }
    const chosen = await openAnalyseDialog(this.ctx, {
      title: `${first ? "Analyse" : "Re-analyse"} ${status.recording_id}`, setup: status.setup.ref, models, runOn,
      submitLabel: first ? "Analyse" : "Re-analyse", dev: this.ctx.dev,
      warning: first ? null : "Re-analysing replaces every pose, including your fixes. Mask edits are kept.",
    });
    if (!chosen) return;
    const {models: refs, mask, body, ...dev} = chosen;
    try {
      await post("/api/analyse", {path: status.recording, models: refs, ...runOn.payload(), ...dev});
      this.ctx.toast(`Analysing ${status.recording_id}`, "ok");
      this.preview = null; this.mode = "idle"; this.timeline.set("preview", null);
      await this.loadStatus();
      await this.loadIssues();
      this.renderFix();
    } catch (error) { this.ctx.toast(error.message, "error"); }
  }

  // ---------------------------------------------------------------- frames

  seekFrame(frame) { if (this.status && Number.isFinite(frame)) this.seek(this.rowOf(frame)); }

  step(delta) { this.seek(Math.max(0, Math.min(this.rows - 1, this.row + delta))); }

  seek(row, {settle = true} = {}) {
    if (!this.rows) return;
    row = Math.max(0, Math.min(this.rows - 1, row));
    if (this.mask.active && row !== this.row) {
      if (this.mask.dirty && !window.confirm(`Discard your mask edit on frame ${this.mask.frame}?`)) return;
      this.stopMaskEdit();
    }
    this.row = row;
    this.timeline.setRow(row);
    this.frameInput.value = this.frameOf(row);
    this.loadFrame(row, "light").then(() => {
      if (!settle) return;
      clearTimeout(this.settleTimer);
      this.settleTimer = setTimeout(() => { if (!this.playing && this.row === row) this.loadFrame(row, "full"); }, SETTLE_MS);
    });
  }

  async loadFrame(row, detail) {
    const token = ++this.generation, frame = this.frameOf(row);
    let payload;
    try {
      payload = await api(`${this.base}/frame${query({frame, detail, raw: this.dev?.raw ? 1 : null})}`);
    } catch (error) {
      if (token !== this.generation) return;
      if (this.current) this.renderStatusLine(`could not load: ${error.message}`);
      else this.showEmpty(`Frame ${frame} could not be read: ${error.message}`);
      return;
    }
    if (token !== this.generation) return;
    const image = await loadImage(this.dev?.raw && payload.image_raw ? payload.image_raw : payload.layers.image).catch(() => null);
    if (token !== this.generation) return;
    if (!image) { this.showEmpty(payload.errors?.[0] || `Frame ${frame} has no image`); return; }
    this.empty.hidden = true;
    this.current = payload;
    if (detail === "full") {
      const mask = payload.layers.mask_final ? await decodeGray(payload.layers.mask_final) : null;
      if (token !== this.generation) return;
      this.maskOverlay = mask ? maskCanvas(mask.data, mask.width, mask.height, [58, 160, 255], 0.38) : null;
      this.dev?.frameLoaded(payload);
      if (this.layers.ap && this.status.has_body_model) this.loadAp(frame, token);
    } else {
      this.maskOverlay = null;
      if (!this.apCache.has(frame)) this.apCanvas = null;
    }
    if (this.apCache.has(frame)) this.apCanvas = this.apCache.get(frame);
    this.canvas.setImage(image);
    this.renderStatusLine();
    this.renderPreviewNumbers();
  }

  async loadAp(frame, token) {
    if (this.apCache.has(frame)) { this.apCanvas = this.apCache.get(frame); this.canvas.redraw(); return; }
    try {
      const fields = await api(`${this.base}/network-fields${query({frame})}`);
      const canvas = colorize(await decodeGray(fields.ap), AP_LUT);
      if (this.apCache.size > 32) this.apCache.delete(this.apCache.keys().next().value);
      this.apCache.set(frame, canvas);
      this.dev?.fieldsLoaded(frame, fields);
      if (token === this.generation) { this.apCanvas = canvas; this.canvas.redraw(); }
    } catch (error) {
      if (token === this.generation) this.ctx.toast(`A-P field: ${error.message}`, "error");
    }
  }

  showEmpty(message) {
    this.empty.textContent = message;
    this.empty.hidden = false;
  }

  toggleLayer(id) {
    if (id === "ap" && !this.status?.has_body_model) return;
    this.layers[id] = !this.layers[id];
    this.canvas.setLayerVisible(id, this.layers[id]);
    this.renderLayers();
    if (id === "ap" && this.layers.ap) this.loadAp(this.frameOf(this.row), this.generation);
  }

  renderLayers() {
    for (const button of this.layerButtons) {
      const id = button.dataset.layer;
      button.setAttribute("aria-pressed", String(this.layers[id]));
      button.hidden = id === "ap" && !this.status?.has_body_model;
    }
  }

  // ------------------------------------------------------------ playback

  togglePlay() { if (this.playing) this.stopPlay(); else this.play(); }

  stopPlay() {
    if (!this.playing) return;
    this.playing = false;
    this.playButton.textContent = "Play";
    this.seek(this.row);
  }

  async play() {
    if (!this.rows || this.mode === "mask") return;
    this.playing = true;
    this.playButton.textContent = "Pause";
    const [lo, hi] = this.loop && this.target ? [this.target.first, this.target.last] : [0, this.rows - 1];
    if (this.row < lo || this.row >= hi) this.row = lo;
    while (this.playing) {
      const started = performance.now();
      this.timeline.setRow(this.row);
      this.frameInput.value = this.frameOf(this.row);
      await this.loadFrame(this.row, "light");
      const interval = 1000 * this.status.step / (this.fps * this.speed), spent = performance.now() - started;
      if (spent < interval) await sleep(interval - spent);
      if (!this.playing) break;
      let next = this.row + Math.max(1, Math.round(Math.max(spent, interval) / interval));
      if (next > hi) {
        if (!this.loop) { this.row = hi; this.stopPlay(); break; }
        next = lo;
      }
      this.row = next;
    }
  }

  // --------------------------------------------------------------- issues

  issueAt(row) {
    return (this.issues?.issues || []).find((issue) => issue.rows[0] <= row && row <= issue.rows[1]) || null;
  }

  // The target of the fixes: an issue, a dragged range, or null (the current frame).
  select(target, {seek = true} = {}) {
    if (this.mode === "relabel" || this.mode === "preview" || this.mode === "job" || this.mode === "mask") {
      if (target?.kind === "range" && this.mode === "relabel") return;
    }
    this.target = target;
    this.selectedIssue = target?.kind === "issue" ? target.issue : null;
    this.timeline.set("selection", target?.kind === "range" ? [target.first, target.last] : null);
    this.timeline.set("selected", this.selectedIssue);
    if (target && seek) {
      this.timeline.showRows(target.first, target.last);
      if (target.kind === "issue") this.seek(target.first);
    }
    this.selectionNote.replaceChildren(...(target?.kind === "range"
      ? [el("span", {}, `Selected ${range([this.frameOf(target.first), this.frameOf(target.last)])}`),
        el("button", {class: "link", onclick: () => this.select(null)}, "Clear")]
      : []));
    this.renderIssues();
    this.renderFix();
  }

  selectIssue(issue, options) {
    this.select(issue ? {first: issue.rows[0], last: issue.rows[1], kind: "issue", issue: issue.id} : null, options);
    this.issuesNode.querySelector(`[data-issue="${CSS.escape(issue?.id || "")}"]`)?.scrollIntoView({block: "nearest"});
  }

  moveIssue(delta, {unreviewed = false} = {}) {
    const list = this.issues?.issues || [];
    if (!list.length) return;
    const index = list.findIndex((issue) => issue.id === this.selectedIssue);
    let at = index < 0 ? (delta > 0 ? -1 : list.length) : index;
    for (let k = 0; k < list.length; k++) {
      at += delta;
      if (at < 0 || at >= list.length) return;
      if (!unreviewed || list[at].state === "unreviewed") { this.selectIssue(list[at]); return; }
    }
  }

  async looksOk() {
    const target = this.target;
    if (!target || !this.issues || this.busy) return;
    this.busy = true;
    try {
      this.issues = await post(`${this.base}/issues/review`, {first: this.frameOf(target.first), last: this.frameOf(target.last), revision: this.issues.revision});
      this.syncIssues();
      const list = this.issues.issues;
      const next = list.find((issue) => issue.state === "unreviewed" && issue.rows[0] > target.last) || list.find((issue) => issue.state === "unreviewed");
      if (next) this.selectIssue(next); else { this.select(null); this.ctx.toast("Every issue is reviewed", "ok"); }
    } catch (error) {
      if (error.status === 409) await this.loadIssues();
      this.ctx.toast(error.message, "error");
    } finally { this.busy = false; }
  }

  renderIssues() {
    const node = this.issuesNode, status = this.status;
    if (!status) { node.replaceChildren(); return; }
    const heading = (text, extra = null) => el("div", {class: "ws-panel-head"}, el("h3", {}, text), extra);
    if (["queued", "analysing"].includes(status.state)) {
      const [text, , progress] = describeStatus(status);
      node.replaceChildren(heading("Issues"), el("div", {class: "ws-analysing"}, el("p", {}, text),
        el("progress", {max: 1, value: progress ?? 0}), el("p", {class: "note"}, "Issues appear when the analysis finishes. You can close the app meanwhile.")));
      return;
    }
    if (!status.analysed) {
      const failed = status.state === "failed";
      node.replaceChildren(heading("Issues"), el("div", {class: "ws-analysing"},
        el("p", {}, failed ? "The analysis failed." : "This recording has not been analysed yet."),
        failed && status.analysis?.error ? el("pre", {class: "ws-error"}, status.analysis.error.split("\n").slice(-4).join("\n")) : null,
        el("button", {class: "primary", onclick: () => (failed ? this.retryAnalysis() : this.reanalyse(true))}, failed ? "Retry" : "Analyse")));
      return;
    }
    if (this.issuesError) {
      node.replaceChildren(heading("Issues"), el("p", {class: "note"}, this.issuesError.status === 409 ? "A job is writing this workspace; the issues come back when it ends." : this.issuesError.message));
      return;
    }
    if (!this.issues) { node.replaceChildren(heading("Issues"), el("p", {class: "note"}, "Loading…")); return; }
    const {summary, issues} = this.issues;
    const head = heading("Issues", el("span", {class: "ws-count"}, summary.issues ? `${summary.done} of ${summary.issues} done` : ""));
    if (!issues.length) {
      node.replaceChildren(head, el("p", {class: "ws-all-clear"}, "No issues found. Play through the recording to check, then export."));
      return;
    }
    const bar = el("div", {class: "ws-done-bar"}, el("span", {style: `width:${(100 * summary.done / summary.issues).toFixed(1)}%`}));
    const items = issues.map((issue) => el("div", {
      class: `item ws-issue ${issue.state}`, dataset: {issue: issue.id}, "aria-current": String(issue.id === this.selectedIssue),
      onclick: () => this.selectIssue(issue),
    }, el("span", {class: "ws-issue-dot", title: STATE_WORDS[issue.state]}),
    el("span", {class: "ws-issue-text"}, el("span", {class: "ws-issue-frames"}, range(issue.frames)), el("span", {class: "ws-issue-reasons"}, issue.reasons.join(", ") || "placed by hand")),
    el("span", {class: "ws-issue-state"}, STATE_WORDS[issue.state])));
    const target = this.target;
    const nav = el("div", {class: "ws-issue-nav"},
      el("button", {type: "button", title: "Previous issue (↑)", onclick: () => this.moveIssue(-1)}, "◀ Prev"),
      el("button", {type: "button", class: "primary", disabled: !target, title: "Mark reviewed and go to the next issue (O)", onclick: () => this.looksOk()}, "Looks OK"),
      el("button", {type: "button", title: "Next issue (↓)", onclick: () => this.moveIssue(1)}, "Next ▶"));
    node.replaceChildren(head, bar, el("div", {class: "list ws-issue-list"}, items), nav);
  }

  // ------------------------------------------------------------- the fixes

  targetFrames() {
    const t = this.target;
    return t ? [this.frameOf(t.first), this.frameOf(t.last)] : [this.frameOf(this.row), this.frameOf(this.row)];
  }

  renderFix() {
    const node = this.fixNode;
    if (!this.status?.analysed || ["queued", "analysing"].includes(this.status.state)) { node.replaceChildren(); return; }
    const head = (text) => el("div", {class: "ws-panel-head"}, el("h3", {}, text));
    if (this.mode === "job") {
      const job = this.pending.job;
      node.replaceChildren(head(this.pending.title), el("div", {class: "ws-job"},
        el("progress", {max: 1, value: job?.progress || 0}),
        el("p", {class: "note"}, job?.state === "queued" ? "Waiting to start…" : (job?.message || "Starting…")),
        el("button", {type: "button", onclick: () => this.cancelJob()}, "Cancel")));
      return;
    }
    if (this.mode === "preview") { node.replaceChildren(head(this.preview.title), this.previewBody()); return; }
    if (this.mode === "mask") { node.replaceChildren(head(`Edit mask · frame ${this.mask.frame}`), this.maskBody()); return; }
    if (this.mode === "relabel") { node.replaceChildren(head(`Relabel ${range(this.targetFrames())}`), this.relabelBody()); return; }
    const frames = this.targetFrames(), single = !this.target;
    const what = single ? `frame ${frames[0]}` : `${this.target.kind === "issue" ? "issue" : "selection"} ${range(frames)}`;
    node.replaceChildren(
      head(`Fix ${what}`),
      el("div", {class: "ws-fix-grid"},
        el("button", {type: "button", disabled: single, title: single ? "Select an issue or drag on the timeline" : "Swap head and tail on every frame of the target (F)", onclick: () => this.flip(false)}, "Flip head/tail"),
        el("button", {type: "button", title: "Swap head and tail on this frame only (Shift+F)", onclick: () => this.flip(true)}, "Flip this frame"),
        el("button", {type: "button", title: "Fit again from the good frames on either side, then compare (R)", onclick: () => this.refit()}, "Refit"),
        el("button", {type: "button", title: "Paint this frame's mask, then refit around it (E)", onclick: () => this.startMaskEdit()}, "Edit mask"),
        el("button", {type: "button", class: "ws-wide", disabled: single, title: single ? "Select an issue or drag on the timeline" : "Label a few keyframes by hand and fit the frames between them (L)", onclick: () => this.startRelabel()}, "Relabel…"),
      ),
      ...(this.dev ? [this.dev.refitOverride()] : []),
    );
  }

  async flip(frameOnly) {
    if (this.busy) return;
    if (!frameOnly && !this.target) return;
    const [first, last] = this.targetFrames(), frame = this.frameOf(this.row);
    this.busy = true;
    try {
      const response = await post(`${this.base}/fixes/flip`, frameOnly ? {frame} : {first, last, frame});
      this.ctx.toast(frameOnly ? `Flipped frame ${frame}` : `Flipped ${range([first, last])}`, "ok");
      await this.changed(response);
    } catch (error) { this.ctx.toast(error.message, "error"); } finally { this.busy = false; }
  }

  async refit({keep = false, frames = null} = {}) {
    if (this.mode !== "idle" && !keep) return;
    const [first, last] = frames || this.targetFrames();
    try {
      const answer = await post(`${this.base}/fixes/refit`, {first, last, keep, ...(this.dev?.refitParams() || {})});
      const span = range(answer.plan.frames);
      await this.follow({kind: "refit", job: answer.job, preview: answer.preview, keep,
        title: keep ? `Refitting ${span} after the mask edit` : `Refitting ${span}`, done: keep ? `Refit ${span} kept` : null});
    } catch (error) { this.ctx.toast(error.message, "error"); }
  }

  async follow(pending) {
    this.pending = pending;
    this.mode = "job";
    this.renderFix();
    while (this.pending === pending) {
      try { pending.job = await api(`/api/jobs/${pending.job.id}`); } catch (error) { this.ctx.toast(error.message, "error"); break; }
      if (this.pending !== pending) return;
      this.renderFix();
      if (["done", "failed", "cancelled"].includes(pending.job.state)) break;
      await sleep(JOB_POLL_MS);
    }
    if (this.pending !== pending) return;
    this.pending = null;
    this.mode = "idle";
    const job = pending.job;
    if (job.state === "done" && pending.keep) {
      this.ctx.toast(pending.done, "ok");
      await this.changed();
    } else if (job.state === "done") {
      await this.showPreview(pending.preview);
    } else {
      if (job.state === "failed") this.ctx.toast(`${pending.title} failed: ${(job.error || "").split("\n").filter(Boolean).pop() || "see the job log"}`, "error");
      this.renderFix();
      if (pending.keep) await this.changed();
    }
  }

  async cancelJob() {
    const pending = this.pending;
    if (!pending) return;
    try { await post(`/api/jobs/${pending.job.id}/cancel`); } catch (error) { this.ctx.toast(error.message, "error"); }
  }

  // A preview left from an earlier visit (its job finished while the page was closed).
  async resumePreview() {
    let previews = [];
    try { previews = await api(`${this.base}/fixes/previews`); } catch { return; }
    if (previews.length) await this.showPreview(previews[0].id, {seek: false});
  }

  async showPreview(id, {seek = true} = {}) {
    let payload;
    try { payload = await api(`${this.base}/fixes/previews/${encodeURIComponent(id)}`); } catch (error) { this.ctx.toast(error.message, "error"); this.renderFix(); return; }
    const byFrame = new Map(payload.per_frame.map((entry) => [entry.frame, entry]));
    const verb = payload.kind === "stitch" ? "Relabel" : "Refit";
    this.preview = {...payload, byFrame, title: `${verb} ${range(payload.frames)} · before / after`};
    this.mode = "preview";
    const rows = [this.rowOf(payload.frames[0]), this.rowOf(payload.frames[1])];
    this.timeline.set("preview", rows);
    this.renderFix();
    if (seek) { this.timeline.showRows(rows[0], rows[1]); this.seek(byFrame.has(this.frameOf(this.row)) ? this.row : rows[0]); }
    this.canvas.redraw();
    this.legend.hidden = false;
    this.legend.replaceChildren(el("span", {class: "ws-key before"}, "before"), el("span", {class: "ws-key after"}, "after"));
  }

  previewFrame() {
    if (this.mode !== "preview" || !this.preview || !this.status) return null;
    return this.preview.byFrame.get(this.frameOf(this.row)) || null;
  }

  previewBody() {
    const p = this.preview, before = p.metrics_before || {}, after = p.metrics_after || {};
    const metric = (key, label, digits = 3) => (before[key] !== undefined || after[key] !== undefined
      ? el("tr", {}, el("td", {}, label), el("td", {class: "num"}, fmt(before[key], digits)), el("td", {class: "num"}, fmt(after[key], digits))) : null);
    this.previewNumbers = el("p", {class: "note ws-preview-frame"});
    const body = el("div", {class: "ws-preview"},
      el("p", {class: "note"}, `${plural(p.frames_placed, "frame")} would change${p.reasons?.length ? ` · ${p.reasons.join(", ")}` : ""}.`),
      el("table", {class: "data ws-metrics"}, el("thead", {}, el("tr", {}, el("th", {}), el("th", {class: "num"}, "Before"), el("th", {class: "num"}, "After"))),
        el("tbody", {}, metric("median_iou", "Overlap (median)"), metric("min_iou", "Overlap (worst)"), metric("orientation_flips", "Head/tail swaps", 0), metric("pose_jumps_over_width", "Jumps", 0))),
      this.previewNumbers,
      p.stale ? el("p", {class: "note warn"}, `Cannot keep: ${p.stale}.`) : null,
      el("div", {class: "ws-row-actions"},
        el("button", {type: "button", class: "primary", disabled: !!p.stale, title: "Use the new poses (Enter)", onclick: () => this.keepPreview()}, "Keep"),
        el("button", {type: "button", title: "Throw the new poses away (Esc)", onclick: () => this.discardPreview()}, "Discard")),
    );
    this.renderPreviewNumbers();
    return body;
  }

  renderPreviewNumbers() {
    if (!this.previewNumbers) return;
    const entry = this.previewFrame();
    this.previewNumbers.textContent = entry
      ? `Frame ${entry.frame}: overlap ${fmt(entry.before?.iou)} → ${fmt(entry.after.iou)}`
      : `Frame ${this.frameOf(this.row)} is outside this fix.`;
  }

  closePreview() {
    this.preview = null; this.previewNumbers = null;
    this.mode = "idle";
    this.timeline.set("preview", null);
    this.legend.hidden = true;
    this.canvas.redraw();
  }

  async keepPreview() {
    if (this.mode !== "preview" || this.preview.stale || this.busy) return;
    this.busy = true;
    try {
      const response = await post(`${this.base}/fixes/previews/${encodeURIComponent(this.preview.id)}/keep`, {frame: this.frameOf(this.row)});
      this.ctx.toast(`Kept ${this.preview.title.split(" · ")[0]}`, "ok");
      this.closePreview();
      await this.changed(response);
    } catch (error) { this.ctx.toast(error.message, "error"); } finally { this.busy = false; }
  }

  async discardPreview() {
    if (this.mode !== "preview" || this.busy) return;
    this.busy = true;
    try {
      await api(`${this.base}/fixes/previews/${encodeURIComponent(this.preview.id)}`, {method: "DELETE"});
      this.closePreview();
      this.renderFix();
    } catch (error) { this.ctx.toast(error.message, "error"); } finally { this.busy = false; }
  }

  // Edit mask --------------------------------------------------------------

  async startMaskEdit() {
    if (this.mode !== "idle") return;
    this.stopPlay();
    try { await this.mask.start(this.frameOf(this.row)); } catch (error) { this.ctx.toast(error.message, "error"); return; }
    this.mode = "mask";
    this.canvasHost.classList.add("painting");
    this.renderFix();
    this.canvas.redraw();
  }

  stopMaskEdit() {
    this.mask.stop();
    this.canvasHost.classList.remove("painting");
    if (this.mode === "mask") this.mode = "idle";
    this.renderFix();
    this.canvas.redraw();
  }

  maskBody() {
    const m = this.mask;
    const brush = (name, label, key) => el("button", {type: "button", "aria-pressed": String(m.brushName === name), title: `${label} brush (${key})`,
      onclick: () => { m.setBrush(name); this.renderFix(); }}, label);
    const size = el("input", {type: "range", min: 2, max: 80, value: m.diameter, "aria-label": "Brush size",
      oninput: (event) => { m.diameter = Number(event.target.value); sizeLabel.textContent = `${m.diameter}px`; this.canvas.redraw(); }});
    const sizeLabel = el("span", {class: "note"}, `${m.diameter}px`);
    return el("div", {class: "ws-mask-tools"},
      el("div", {class: "ws-row-actions"}, brush("worm", "Worm", "W"), brush("background", "Background", "B")),
      el("label", {class: "ws-size"}, "Size", size, sizeLabel),
      el("div", {class: "ws-row-actions"},
        el("button", {type: "button", disabled: !m.history.length, title: "Undo the last stroke (Ctrl+Z)", onclick: () => { if (m.undo()) { this.canvas.redraw(); this.renderFix(); } }}, "Undo"),
        el("button", {type: "button", disabled: !m.hasProposal, title: "Replace the draft with the network's mask (N)", onclick: () => { if (m.useProposal()) { this.canvas.redraw(); this.renderFix(); } }}, "Network proposal")),
      el("p", {class: "note"}, "Paint with the left button; Shift-drag or right-drag pans. Saving refits the frames around this one and keeps the result."),
      el("div", {class: "ws-row-actions"},
        el("button", {type: "button", class: "primary", disabled: !m.dirty, title: "Save and refit (Enter)", onclick: () => this.saveMask()}, "Save & refit"),
        el("button", {type: "button", title: "Leave without saving (Esc)", onclick: () => this.cancelMask()}, "Cancel")),
    );
  }

  cancelMask() {
    if (this.mask.dirty && !window.confirm("Discard your mask edit?")) return;
    this.stopMaskEdit();
  }

  async saveMask() {
    if (this.mode !== "mask" || !this.mask.dirty || this.busy) return;
    this.busy = true;
    const frame = this.mask.frame, issue = this.issueAt(this.rowOf(frame));
    try {
      const response = await this.mask.save();
      this.stopMaskEdit();
      await this.changed(response);
      // Refit the issue the frame is in, else the frame (its anchors widen it to the stretch around it).
      await this.refit({keep: true, frames: issue ? issue.frames : [frame, frame]});
    } catch (error) { this.ctx.toast(error.message, "error"); } finally { this.busy = false; }
  }

  // Relabel ----------------------------------------------------------------

  async startRelabel() {
    if (this.mode !== "idle" || !this.target) return;
    this.keyframes = {spacing: this.keyframes?.spacing || 10, rows: []};
    this.mode = "relabel";
    await this.proposeKeyframes();
  }

  async proposeKeyframes() {
    const [first, last] = this.targetFrames();
    try {
      const answer = await api(`${this.base}/fixes/keyframes${query({first, last, spacing: this.keyframes.spacing})}`);
      this.keyframes.rows = answer.frames.map((frame) => this.rowOf(frame));
    } catch (error) { this.ctx.toast(error.message, "error"); }
    this.timeline.set("keyframes", this.keyframes.rows);
    this.renderFix();
  }

  toggleKeyframe(row) {
    if (this.mode !== "relabel") return;
    const rows = this.keyframes.rows, [a, b] = this.timeline.window;
    const near = Math.max(1, Math.round((b - a) / this.timelineHost.clientWidth * 5));
    const hit = rows.findIndex((r) => Math.abs(r - row) <= near);
    if (hit >= 0) rows.splice(hit, 1); else { rows.push(row); rows.sort((x, y) => x - y); }
    this.timeline.set("keyframes", [...rows]);
    this.seek(row);
    this.renderFix();
  }

  relabelBody() {
    const k = this.keyframes;
    const spacing = el("input", {type: "number", min: 1, max: 500, value: k.spacing, class: "ws-spacing", "aria-label": "Keyframe spacing",
      onchange: (event) => { k.spacing = Math.max(1, Number(event.target.value) || 10); this.proposeKeyframes(); }});
    return el("div", {class: "ws-relabel"},
      el("label", {class: "ws-form-row"}, "One keyframe every", spacing, "frames"),
      el("p", {class: "note"}, `${plural(k.rows.length, "keyframe")}. Click the timeline to add or remove one. Each gets a full label in Labeling; the frames between them are then fitted from those labels.`),
      el("div", {class: "ws-row-actions"},
        el("button", {type: "button", class: "primary", disabled: !k.rows.length, title: "Open the keyframes in Labeling (Enter)", onclick: () => this.sendRelabel()}, `Label ${plural(k.rows.length, "keyframe")}`),
        el("button", {type: "button", title: "Esc", onclick: () => this.stopRelabel()}, "Cancel")),
    );
  }

  stopRelabel() {
    this.mode = "idle";
    this.timeline.set("keyframes", null);
    this.renderFix();
  }

  async sendRelabel() {
    if (this.mode !== "relabel" || !this.keyframes.rows.length) return;
    try {
      const queue = await post("/api/queues", {kind: "relabel", workspace: this.name, frames: this.keyframes.rows.map((row) => this.frameOf(row))});
      this.stopRelabel();
      this.ctx.navigate("labeling", `queue/${encodeURIComponent(queue.id)}`);
    } catch (error) { this.ctx.toast(`Could not open the keyframes in Labeling: ${error.message}`, "error"); }
  }

  // Back from Labeling with the queue's labels: stitch the stretch between them.
  async stitch(queue) {
    if (this.mode !== "idle") { this.ctx.toast("Finish the current fix before stitching the relabeled frames.", "error"); return; }
    try {
      const answer = await post(`/api/queues/${encodeURIComponent(queue)}/stitch`);
      const span = range(answer.plan.frames);
      await this.follow({kind: "stitch", job: answer.job, preview: answer.preview, title: `Fitting ${span} between the keyframes`});
    } catch (error) { this.ctx.toast(`Could not stitch the relabeled frames: ${error.message}`, "error"); }
  }

  // Fixes list ------------------------------------------------------------

  renderFixes() {
    const node = this.fixesNode;
    if (!this.status?.analysed) { node.replaceChildren(); return; }
    const head = el("div", {class: "ws-panel-head"}, el("h3", {}, "Fixes"), el("span", {class: "ws-count"}, this.fixes.length ? String(this.fixes.length) : ""));
    if (!this.fixes.length) { node.replaceChildren(head, el("p", {class: "note"}, "Nothing changed by hand yet.")); return; }
    node.replaceChildren(head, el("div", {class: "list ws-fix-list"}, this.fixes.map((fix) => el("div", {class: "item ws-fix-item", onclick: () => fix.frames && this.seekFrame(fix.frames[0])},
      el("span", {class: "ws-fix-text"}, el("span", {}, fix.title), el("span", {class: "note"}, fix.frames ? range(fix.frames) : "")),
      el("button", {type: "button", class: "ws-undo", disabled: !fix.undoable,
        title: fix.blocked_by ? `Undo ${this.fixes.find((f) => f.id === fix.blocked_by)?.title || fix.blocked_by} first: it changed the same frames later` : "Undo this fix",
        onclick: (event) => { event.stopPropagation(); this.undo(fix); }}, "Undo")))));
  }

  async undo(fix) {
    if (this.busy) return;
    this.busy = true;
    try {
      const response = await post(`${this.base}/fixes/${encodeURIComponent(fix.id)}/undo`, {frame: this.frameOf(this.row)});
      this.ctx.toast(`Undid: ${fix.title}`, "ok");
      await this.changed(response);
    } catch (error) { this.ctx.toast(error.message, "error"); } finally { this.busy = false; }
  }

  // Frame status line -----------------------------------------------------

  renderStatusLine(problem = null) {
    if (!this.status) return;
    const frame = this.frameOf(this.row), payload = this.current, issue = this.issueAt(this.row);
    const parts = [el("strong", {}, `Frame ${frame}`)];
    const provenance = payload?.provenance;
    const byHand = provenance && (provenance.edit || /^(fix|candidates):/.test(provenance.job || ""));
    let kind = "ok", text = "OK";
    if (!this.status.analysed) { kind = ""; text = "not analysed"; }
    else if (payload && !payload.pose) { kind = "warn"; text = "no pose"; }
    else if (issue && issue.state === "unreviewed") { kind = "warn"; text = `${issue.reasons.join(", ")} · to review`; }
    else if (byHand) { kind = "fixed"; text = "fixed by hand"; }
    else if (issue) { kind = issue.state === "fixed" ? "fixed" : "ok"; text = `${issue.reasons.join(", ") || "placed by hand"} · ${STATE_WORDS[issue.state]}`; }
    if (payload?.mask_stale) text += " · mask edited, pose not refit";
    if (problem) { kind = "error"; text = problem; }
    parts.push(el("span", {class: `ws-status-dot ${kind}`}), el("span", {}, text));
    this.frameStatus.replaceChildren(...parts);
  }

  // Export ----------------------------------------------------------------

  async export() {
    const status = this.status, unreviewed = this.issues?.summary?.unreviewed ?? 0;
    const setup = status.setup || {};
    let previous = [];
    try { previous = await api(`${this.base}/exports`); } catch { /* the list is optional */ }
    const dialog = el("dialog", {class: "ws-dialog ws-export"});
    const result = el("div", {class: "ws-export-result"});
    const go = el("button", {class: "primary", type: "button"}, unreviewed ? "Export anyway" : "Export");
    const list = (exports) => (exports.length ? el("div", {class: "ws-export-list"}, el("h3", {}, "Exports"), exports.slice(0, 5).map((e) =>
      el("div", {class: "ws-export-row"}, el("a", {href: e.download_url, download: ""}, e.table), el("span", {class: "note"}, new Date(e.created_at).toLocaleString()),
        el("a", {href: e.metadata_url, class: "note", target: "_blank"}, "export.json")))) : null);
    go.addEventListener("click", async () => {
      go.disabled = true;
      go.textContent = "Exporting…";
      try {
        const done = await post(`${this.base}/export`, {pixel_size_um: setup.pixel_size_um ?? null, fps: setup.fps ?? null, setup: setup.ref ?? null});
        result.replaceChildren(el("p", {}, "Exported. ", el("a", {href: done.download_url, download: ""}, `Download ${done.table}`)));
        previous = [done, ...previous];
        listNode.replaceChildren(...[list(previous)].filter(Boolean));
        go.hidden = true;
        cancel.textContent = "Close";
        this.loadStatus();
      } catch (error) {
        result.replaceChildren(el("p", {class: "note error"}, error.message));
        go.disabled = false; go.textContent = "Try again";
      }
    });
    const cancel = el("button", {type: "button", onclick: () => dialog.close()}, "Cancel");
    const listNode = el("div", {}, list(previous));
    const units = setup.pixel_size_um ? `${setup.pixel_size_um} µm per pixel` : "pixels (the setup has no pixel size)";
    dialog.append(
      el("h2", {}, `Export ${status.recording_id}`),
      unreviewed ? el("p", {class: "ws-export-warn"}, `${plural(unreviewed, "issue")} not reviewed — export anyway?`) : el("p", {}, "Every issue is reviewed."),
      el("p", {class: "note"}, `One table per frame: midline, curvature, width, head/tail, velocity and status. Lengths in ${units}${setup.fps ? `, ${setup.fps} fps` : ""}.`),
      result, listNode,
      el("div", {class: "ws-dialog-actions"}, cancel, go),
    );
    dialog.addEventListener("close", () => dialog.remove());
    document.body.append(dialog);
    dialog.showModal();
    go.focus();
  }

  // Help and keyboard ------------------------------------------------------

  help() {
    const rows = [
      ["Space", "Play / pause"], ["← →", "Previous / next frame (Shift: 10 frames)"], ["↑ ↓", "Previous / next issue"],
      ["O", "Looks OK: mark the issue reviewed, go to the next"], ["F", "Flip head/tail on the issue or selection"], ["Shift+F", "Flip this frame"],
      ["R", "Refit"], ["E", "Edit mask"], ["L", "Relabel"], ["Enter / Esc", "Keep / Discard a preview; Save / Cancel an edit"],
      ["W  B", "Worm / Background brush"], ["[  ]", "Smaller / larger brush"], ["N", "Network proposal (while editing)"], ["Ctrl+Z", "Undo a stroke"],
      ["1 2 3 4", "Mask, Midline, Outline, A-P field"], ["Esc", "Clear the selection"], ["Wheel / drag", "Zoom / select on the timeline; zoom / pan the frame (Shift-drag)"],
    ];
    const dialog = el("dialog", {class: "ws-dialog ws-help"},
      el("h2", {}, "Keyboard shortcuts"),
      el("table", {class: "data"}, el("tbody", {}, rows.map(([key, what]) => el("tr", {}, el("td", {}, key.split("  ").map((k) => el("kbd", {}, k))), el("td", {}, what))))),
      el("div", {class: "ws-dialog-actions"}, el("button", {type: "button", onclick: () => dialog.close()}, "Close")));
    dialog.addEventListener("close", () => dialog.remove());
    document.body.append(dialog);
    dialog.showModal();
  }

  _key(event) {
    if (this.node.hidden || !this.status || document.querySelector("dialog[open]")) return;
    const target = event.target;
    if (target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName))) return;
    if (event.altKey || ((event.ctrlKey || event.metaKey) && !(this.mode === "mask" && event.key.toLowerCase() === "z"))) return;
    const key = event.key, lower = key.toLowerCase();
    const handled = () => event.preventDefault();
    if (this.mode === "mask") {
      if (lower === "z" && (event.ctrlKey || event.metaKey)) { if (this.mask.undo()) { this.canvas.redraw(); this.renderFix(); } return handled(); }
      if (lower === "w" || lower === "b") { this.mask.setBrush(lower === "w" ? "worm" : "background"); this.renderFix(); this.canvas.redraw(); return handled(); }
      if (key === "[" || key === "]") { this.mask.resize(key === "]" ? 2 : -2); this.renderFix(); this.canvas.redraw(); return handled(); }
      if (lower === "n") { if (this.mask.useProposal()) { this.canvas.redraw(); this.renderFix(); } return handled(); }
      if (key === "Enter") { this.saveMask(); return handled(); }
      if (key === "Escape") { this.cancelMask(); return handled(); }
    }
    if (this.mode === "preview") {
      if (key === "Enter") { this.keepPreview(); return handled(); }
      if (key === "Escape") { this.discardPreview(); return handled(); }
    }
    if (this.mode === "relabel") {
      if (key === "Enter") { this.sendRelabel(); return handled(); }
      if (key === "Escape") { this.stopRelabel(); return handled(); }
    }
    switch (key) {
      case " ": this.togglePlay(); return handled();
      case "ArrowLeft": this.step(event.shiftKey ? -10 : -1); return handled();
      case "ArrowRight": this.step(event.shiftKey ? 10 : 1); return handled();
      case "ArrowUp": this.moveIssue(-1); return handled();
      case "ArrowDown": this.moveIssue(1); return handled();
      case "Home": this.seek(this.target ? this.target.first : 0); return handled();
      case "End": this.seek(this.target ? this.target.last : this.rows - 1); return handled();
      case "Escape": if (this.target) { this.select(null); return handled(); } return;
      case "?": this.help(); return handled();
      default: break;
    }
    const layer = LAYERS.find((l) => l.key === key);
    if (layer) { this.toggleLayer(layer.id); return handled(); }
    if (this.mode !== "idle") return;
    if (lower === "o") { this.looksOk(); return handled(); }
    if (lower === "f") { this.flip(event.shiftKey); return handled(); }
    if (lower === "r") { this.refit(); return handled(); }
    if (lower === "e") { this.startMaskEdit(); return handled(); }
    if (lower === "l") { this.startRelabel(); return handled(); }
  }
}

function fmt(value, digits = 3) {
  return value === null || value === undefined || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
}
