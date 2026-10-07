// The frame editor of the Labeling page: one frame's label, mask first, then body, then Save.
//
// The page (labeling.js) owns the queues and navigation; it hands the editor
// a request naming the frame ({setup, dataset, entry, queue}) and asks it to
// save. Everything the editor shows comes from /api/labeling (see
// app/routers/labeling.py):
//
//   mask    painted locally (Worm / Background brush); a proposal (the
//           network's probability or the frame's darkness, thresholded by one
//           slider) previews until Apply; Refine runs on the server; Undo and
//           Revert (to the mask the frame opened with).
//   body    what the label will say about the body: "auto" (nobody chose:
//           the targets builder orients it), "head" (a head end, from Flip or
//           from accepting the shown body) or "trace" (a midline, head first,
//           traced by hand or the proposal's). Its fit, when there is one, is
//           what the Midline and A-P layers draw. The body model's proposal is
//           computed as the frame opens when the server has a GPU; without one
//           (15-20 s a frame on the CPU) only when asked: Propose, or Use
//           proposal (G).
//   context the frames t-16..t+16, loaded on first use: Frames (offset,
//           Play at 10 fps) or Difference (frame[t+lag] - frame[t-lag],
//           contrast set automatically).
//
// Layers: Mask (opacity slider, M toggles), A-P field (the body's, else the
// proposal's), Midline + head/tail (with the acquisition nose), Proposal;
// with --dev also the overlap, the tube outline and the stored trace.

import {el, post} from "./api.js";
import {FrameCanvas, drawMidline} from "./frame_canvas.js";
import {
  BACKGROUND, WORM, apOverlay, applyProposal, decodeMask, differenceCanvas, encodeMask, grayCanvas, grayOf, loadImage,
  maskOverlay, reversedAp, sameMask, stroke, thresholded,
} from "./labeling_raster.js";

const UNDO_LIMIT = 30;
const PLAY_FPS = 10;
const LAYERS = [
  {id: "mask", label: "Mask", key: "M"},
  {id: "ap", label: "A-P"},
  {id: "midline", label: "Midline"},
  {id: "proposal", label: "Proposal"},
  {id: "overlap", label: "Overlap", dev: true},
  {id: "tube", label: "Tube", dev: true},
  {id: "stored", label: "Stored trace", dev: true},
];

const button = (label, onclick, attributes = {}) => el("button", {type: "button", onclick, ...attributes}, label);
const keyHint = (key) => el("kbd", {}, key);

export class FrameEditor {
  constructor(ctx, {center, side, onSaved, onChange}) {
    this.ctx = ctx;
    this.onSaved = onSaved;
    this.onChange = onChange || (() => {});
    this.frame = null;          // the open payload of the current frame
    this.request = null;
    this.busy = 0;
    this.brush = WORM;
    this.size = 12;
    this.threshold = 0.5;
    this.opacity = 0.45;
    this.maskShown = true;
    this.raw = false;
    this.contextMode = "frame";
    this.offset = 0;
    this.lag = 4;
    this.playTimer = null;
    this.generation = 0;
    this._build(center, side);
  }

  // ------------------------------------------------------------------ DOM

  _build(center, side) {
    const node = (tag, attributes, ...children) => el(tag, attributes, ...children);
    this.layerButtons = {};
    const layerBar = node("div", {class: "lb-layers", role: "group", "aria-label": "Layers"},
      ...LAYERS.map((layer) => {
        const toggle = button(layer.label, () => this.toggleLayer(layer.id), {"aria-pressed": "true", class: layer.dev ? "dev-only" : null, title: layer.key ? `${layer.label} (${layer.key})` : layer.label});
        this.layerButtons[layer.id] = toggle;
        return toggle;
      }));
    this.opacityInput = node("input", {type: "range", min: 0, max: 1, step: 0.05, value: this.opacity, "aria-label": "Mask opacity", oninput: (e) => { this.opacity = Number(e.target.value); this.maskShown = true; this._syncLayers(); this.view.redraw(); }});
    this.rawButton = button("Raw", () => this.toggleRaw(), {"aria-pressed": "false", title: "Show the frame as recorded (R)"});
    const toolbar = node("div", {class: "lb-toolbar"}, layerBar, node("label", {class: "inline lb-opacity"}, "Opacity", this.opacityInput), this.rawButton);

    this.canvasBox = node("div", {class: "lb-canvas"});
    this.empty = node("div", {class: "lb-empty empty"}, "Choose a queue, or browse labels.");

    this.modeButtons = {
      frame: button("Frames", () => this.setContextMode("frame"), {"aria-pressed": "true"}),
      difference: button("Difference", () => this.setContextMode("difference"), {"aria-pressed": "false", title: "frame[t+lag] − frame[t−lag] (D)"}),
    };
    this.offsetInput = node("input", {type: "range", min: -16, max: 16, step: 1, value: 0, "aria-label": "Context offset", oninput: (e) => this.setOffset(Number(e.target.value))});
    this.offsetValue = node("span", {class: "lb-value"}, "t");
    this.playButton = button("Play", () => this.togglePlay(), {title: "Play the context (Space)"});
    this.lagInput = node("input", {type: "range", min: 1, max: 16, step: 1, value: this.lag, "aria-label": "Lag", oninput: (e) => this.setLag(Number(e.target.value))});
    this.lagValue = node("span", {class: "lb-value"}, `±${this.lag}`);
    this.offsetGroup = node("span", {class: "lb-group"}, this.offsetInput, this.offsetValue, this.playButton);
    this.lagGroup = node("span", {class: "lb-group", hidden: true}, node("span", {class: "note"}, "lag"), this.lagInput, this.lagValue);
    this.contextNote = node("span", {class: "note"});
    const contextBar = node("div", {class: "lb-context"}, node("span", {class: "lb-group seg"}, this.modeButtons.frame, this.modeButtons.difference), this.offsetGroup, this.lagGroup, this.contextNote);
    this.statusLine = node("div", {class: "lb-status"});
    this.centerBody = node("div", {class: "lb-center", hidden: true}, toolbar, this.canvasBox, contextBar, this.statusLine);
    center.append(this.centerBody, this.empty);

    this.view = new FrameCanvas(this.canvasBox);
    this.view.onPointer = (event) => this._pointer(event);
    this.canvasBox.addEventListener("pointerleave", () => { this.cursor = null; this.view.redraw(); });
    this._installLayers();

    // ---- right panel: 1 Mask, 2 Body, Save
    this.brushButtons = {
      [WORM]: button("Worm", () => this.setBrush(WORM), {"aria-pressed": "true", title: "Paint worm (W)"}),
      [BACKGROUND]: button("Background", () => this.setBrush(BACKGROUND), {"aria-pressed": "false", title: "Paint background (E)"}),
    };
    this.sizeInput = node("input", {type: "range", min: 2, max: 60, step: 1, value: this.size, "aria-label": "Brush size", oninput: (e) => { this.size = Number(e.target.value); this.sizeValue.textContent = `${this.size}px`; }});
    this.sizeValue = node("span", {class: "lb-value"}, `${this.size}px`);
    this.proposalButtons = {
      network: button("Network", () => this.preview("network"), {"aria-pressed": "false", title: "The model's mask (N)"}),
      threshold: button("Threshold", () => this.preview("threshold"), {"aria-pressed": "false", title: "Dark pixels"}),
    };
    this.thresholdInput = node("input", {type: "range", min: 0.05, max: 0.95, step: 0.01, value: this.threshold, "aria-label": "Threshold", oninput: (e) => this.setThreshold(Number(e.target.value))});
    this.thresholdValue = node("span", {class: "lb-value"}, this.threshold.toFixed(2));
    this.applyButton = button("Apply (A)", () => this.applyPreview(), {class: "primary", title: "Take the proposal (A)"});
    this.cancelPreviewButton = button("Cancel", () => this.cancelPreview());
    this.previewRow = node("div", {class: "row", hidden: true}, this.applyButton, this.cancelPreviewButton);
    const refine = node("div", {class: "lb-grid"}, ...[
      ["Fill holes", "fill_holes", "Fill narrow holes inside the worm"], ["Largest", "largest", "Keep only the largest piece"],
      ["Grow", "grow", "Grow the mask by one pixel"], ["Shrink", "shrink", "Shrink the mask by one pixel"],
    ].map(([label, method, title]) => button(label, () => this.refine(method), {title})));
    this.undoButton = button("Undo", () => this.undo(), {title: "Undo (Z)"});
    this.revertButton = button("Revert", () => this.revert(), {title: "Back to the mask the frame opened with"});
    const maskSection = node("section", {class: "section"},
      node("h3", {}, "1 · Mask"),
      node("div", {class: "row"}, this.brushButtons[WORM], this.brushButtons[BACKGROUND]),
      node("div", {class: "row lb-slider"}, node("span", {class: "note"}, "Size"), this.sizeInput, this.sizeValue),
      node("div", {class: "row"}, this.proposalButtons.network, this.proposalButtons.threshold),
      node("div", {class: "row lb-slider"}, node("span", {class: "note"}, "Cut"), this.thresholdInput, this.thresholdValue),
      this.previewRow,
      refine,
      node("div", {class: "row"}, this.undoButton, this.revertButton),
    );

    this.bodyNote = node("p", {class: "note lb-body-note"});
    this.bodyControls = node("div", {class: "lb-body-controls"});
    this.useButton = button("Use proposal", () => this.useProposal(), {class: "primary", title: "Use the model's body (G)"});
    // Without a GPU the proposal is computed only on request (see the header).
    this.proposeButton = button("Propose", () => this.requestProposal(), {title: "Compute the body model's proposal for the current mask (about 15-20 s without a GPU)"});
    this.flipButton = button("Flip head/tail", () => this.flip(), {title: "Swap head and tail (H)"});
    this.traceButton = button("Trace midline", () => this.startTrace(false), {title: "Click from head to tail (T)"});
    this.editButton = button("Edit proposal", () => this.startTrace(true), {title: "Trace, starting from the proposal's points"});
    this.traceNote = node("p", {class: "note"});
    this.fitButton = button("Fit", () => this.fitTrace(), {class: "primary"});
    this.removeButton = button("Remove last", () => this.removeLastPoint());
    this.cancelTraceButton = button("Cancel", () => this.endTrace());
    this.acceptButton = button("Accept", () => this.acceptTrace(), {class: "primary"});
    this.discardButton = button("Discard", () => this.endTrace());
    this.tracePanel = node("div", {class: "lb-trace", hidden: true}, this.traceNote,
      node("div", {class: "row"}, this.fitButton, this.removeButton, this.cancelTraceButton, this.acceptButton, this.discardButton));
    this.maskOnly = node("input", {type: "checkbox", onchange: () => this._changed()});
    this.bodyControls.append(
      node("div", {class: "row"}, this.useButton, this.proposeButton, this.flipButton),
      node("div", {class: "row"}, this.traceButton, this.editButton),
      this.tracePanel,
    );
    const bodySection = node("section", {class: "section"},
      node("h3", {}, "2 · Body"),
      this.bodyNote,
      this.bodyControls,
      node("label", {class: "inline lb-maskonly", title: "The body is unclear: this label trains the mask only"}, this.maskOnly, "Mask only (body unclear)"),
    );

    this.saveNextButton = button("Save & next", () => this.save({next: true}), {class: "primary lb-save", title: "Save and open the next frame (Enter)"});
    this.saveButton = button("Save", () => this.save({next: false}), {title: "Save and stay (S)"});
    this.saveNote = node("p", {class: "note lb-save-note"});
    const saveSection = node("section", {class: "section lb-save-row"},
      node("div", {class: "row"}, this.saveNextButton, this.saveButton),
      node("p", {class: "note lb-keys"}, keyHint("Enter"), " save & next · ", keyHint("S"), " save · ", keyHint("?"), " all keys"),
      this.saveNote);
    this.sideBody = node("div", {class: "lb-side", hidden: true}, maskSection, bodySection, saveSection);
    side.append(this.sideBody);
  }

  _installLayers() {
    const v = this.view;
    v.setLayer("mask", (g) => {
      const overlay = this.previewMask ? this.previewOverlay : this.maskOverlay;
      if (!overlay) return;
      g.globalAlpha = this.previewMask ? Math.max(this.opacity, 0.5) : this.opacity;
      g.drawImage(overlay, 0, 0);
    });
    v.setLayer("ap", (g) => {
      const ap = this._apCanvas();
      if (!ap) return;
      g.globalAlpha = 0.6;
      g.drawImage(ap, 0, 0);
    }, false);
    v.setLayer("overlap", (g) => {
      if (!this.overlapCanvas) return;
      g.globalAlpha = 0.7;
      g.drawImage(this.overlapCanvas, 0, 0);
    }, false);
    v.setLayer("tube", (g, view) => {
      const fit = this.body?.fit;
      if (fit?.centerline_xy) drawTube(g, view, fit.centerline_xy, fit.width_profile);
    }, false);
    v.setLayer("proposal", (g, view) => {
      const p = this.proposal;
      if (p?.status !== "ready") return;
      g.setLineDash([6 / view.scale, 5 / view.scale]);
      drawMidline(g, view, p.centerline_xy, {color: "#6cb4ff", head: "#6cb4ff", tail: "#6cb4ff", width: 2});
    });
    v.setLayer("midline", (g, view) => {
      const body = this.body;
      if (body?.fit?.centerline_xy) drawMidline(g, view, body.fit.centerline_xy, {width: 2.5});
      else if (body?.trace?.length > 1) drawMidline(g, view, body.trace, {width: 2});
      const nose = this.frame?.nose_xy;
      if (nose) drawNose(g, view, nose);
    });
    v.setLayer("stored", (g, view) => {
      const trace = this.frame?.body?.trace_xy;
      if (trace) drawPoints(g, view, trace, "#e0b04d");
    }, false);
    v.setLayer("trace", (g, view) => {
      if (!this.trace) return;
      if (this.trace.fit) drawMidline(g, view, this.trace.fit.centerline_xy, {color: "#e0b04d", head: "#e0b04d", width: 2.5});
      drawPoints(g, view, this.trace.points, "#ffd23c");
    });
    v.setLayer("cursor", (g, view) => {
      if (!this.cursor || this.trace || !this.frame) return;
      g.lineWidth = 1 / view.scale;
      g.strokeStyle = this.brush === WORM ? "#ff46a0" : "#ffffff";
      g.beginPath();
      g.arc(this.cursor.x, this.cursor.y, this.size / 2, 0, Math.PI * 2);
      g.stroke();
    });
    this._syncLayers();
  }

  // ------------------------------------------------------------------ opening

  get open() { return !!this.frame; }

  get dirty() {
    if (!this.frame) return false;
    return !sameMask(this.mask, this.openedMask) || this.bodyChanged || this.maskOnly.checked !== !!this.frame.body.mask_only;
  }

  clear() {
    this.stopPlay();
    this.frame = null;
    this.request = null;
    this.centerBody.hidden = true;
    this.sideBody.hidden = true;
    this.empty.hidden = false;
    this.onChange();
  }

  async load(request) {
    const generation = ++this.generation;
    this.stopPlay();
    this.request = request;
    this._status("Opening the frame…");
    const frame = await post("/api/labeling/open", request);
    const [image, raw, gray, mask] = await Promise.all([loadImage(frame.image), loadImage(frame.image_raw), grayOf(frame.image), decodeMask(frame.mask)]);
    if (generation !== this.generation) return false;
    this.frame = frame;
    this.width = frame.width; this.height = frame.height;
    this.images = {flat: image, raw};
    this.gray = gray;
    this.probability = frame.probability ? await grayOf(frame.probability) : null;
    this.mask = mask;
    this.openedMask = mask.slice();
    this.history = [];
    this.previewMask = null;
    this.previewSource = null;
    this.context = null;
    this.contextMode = "frame";
    this.offset = 0;
    this.trace = null;
    this.proposal = null;
    this.proposalMask = null;
    this.body = await this._initialBody(frame);
    this.bodyChanged = false;
    this.maskOnly.checked = !!frame.body.mask_only;
    this.overlapCanvas = frame.targets?.overlap ? await overlapCanvasOf(frame.targets.overlap, this.width, this.height) : null;
    this.empty.hidden = true;
    this.centerBody.hidden = false;
    this.sideBody.hidden = false;
    this.onChange();  // the page shows the tool panel before the frame is fitted to the view
    this.offsetInput.min = -frame.max_lag; this.offsetInput.max = frame.max_lag;
    this.lagInput.max = frame.max_lag;
    this.lag = Math.min(this.lag, frame.max_lag);
    this._status("");
    this._renderMask();
    this._renderBase();
    this.view.fit();
    this._syncContext();
    this._syncControls();
    this.onChange();
    if (frame.models.body && this.autoPropose) this.requestProposal();
    return true;
  }

  async _initialBody(frame) {
    const targets = frame.targets;
    const fit = targets?.centerline_xy ? {...targets, fit_iou: targets.meta?.fit_iou, apValues: await grayOf(targets.ap)} : null;
    if (frame.body.trace_xy) return {kind: "trace", trace: frame.body.trace_xy, head: null, fit};
    if (frame.body.head_xy) return {kind: "head", trace: null, head: frame.body.head_xy, fit};
    return {kind: "auto", trace: null, head: null, fit};
  }

  // ------------------------------------------------------------------ status and controls

  _status(message, kind = "") {
    this.saveNote.textContent = message;
    this.saveNote.className = `note lb-save-note ${kind}`;
  }

  _changed() {
    this._syncControls();
    this.onChange();
  }

  _describe() {
    const f = this.frame;
    if (!f) return "";
    const parts = [f.entry.recording, `frame ${f.entry.frame}`];
    parts.push(f.split ? `split ${f.split}` : f.split_note ? `split ${f.split_note}` : "");
    if (f.label) parts.push(`saved revision ${f.label.revision}${f.label.dataset !== f.saving.dataset ? ` (${f.label.dataset})` : ""}`);
    else parts.push({workspace: "mask from the workspace", network: "mask from the model", empty: "no mask yet"}[f.mask_source] || "");
    if (this.dirty) parts.push("unsaved changes");
    return parts.filter(Boolean).join(" · ");
  }

  _syncControls() {
    if (!this.frame) return;
    this.statusLine.textContent = this._describe();
    for (const [value, b] of Object.entries(this.brushButtons)) b.setAttribute("aria-pressed", String(Number(value) === this.brush));
    for (const [name, b] of Object.entries(this.proposalButtons)) b.setAttribute("aria-pressed", String(this.previewSource === name));
    this.proposalButtons.network.disabled = !this.frame.models.mask;
    this.proposalButtons.network.title = this.frame.models.mask ? `The model's mask: ${this.frame.models.mask} (N)` : "This setup has no mask model yet";
    this.previewRow.hidden = !this.previewMask;
    this.undoButton.disabled = !this.history.length;
    this.revertButton.disabled = sameMask(this.mask, this.openedMask);
    const tracing = !!this.trace, fitted = !!this.trace?.fit;
    this.tracePanel.hidden = !tracing;
    this.fitButton.hidden = this.removeButton.hidden = this.cancelTraceButton.hidden = fitted;
    this.acceptButton.hidden = this.discardButton.hidden = !fitted;
    if (tracing) {
      const n = this.trace.points.length;
      this.traceNote.textContent = fitted ? `Fit IoU ${this.trace.fit.fit_iou.toFixed(3)}. Accept it as the body?`
        : `Click along the midline from head to tail (${n} point${n === 1 ? "" : "s"}). Backspace removes the last, Enter fits.`;
      this.fitButton.disabled = n < 2;
      this.removeButton.disabled = !n;
    }
    const p = this.proposal;
    const onRequest = !this.autoPropose && !!this.frame?.models.body;
    this.proposeButton.hidden = !onRequest;
    this.proposeButton.disabled = tracing || p?.status === "loading" || (p?.status === "ready" && !p.stale);
    this.useButton.disabled = tracing || !(p?.status === "ready" || p?.status === "loading" || onRequest);
    this.editButton.disabled = tracing || p?.status !== "ready";
    this.traceButton.disabled = tracing;
    this.flipButton.disabled = tracing;
    this.bodyControls.classList.toggle("lb-muted", this.maskOnly.checked);
    this.bodyNote.textContent = this._bodyText();
    this.saveNextButton.disabled = this.saveButton.disabled = this.busy > 0 || tracing;
  }

  _bodyText() {
    const body = this.body, p = this.proposal;
    if (this.maskOnly.checked) return "Mask only: the body is left out of this label.";
    let text;
    if (body.kind === "trace") text = `Body: traced${body.fit?.fit_iou != null ? `, fit IoU ${body.fit.fit_iou.toFixed(3)}` : ""}.`;
    else if (body.kind === "head") text = "Body: head end chosen.";
    else if (body.fit) text = "Body: fitted automatically; saving accepts it.";
    else text = "Body: automatic.";
    if (!this.frame.models.body) return `${text} No body model for this setup.`;
    if (p?.status === "loading") return `${text} Proposal: computing…`;
    if (p?.status === "ready") return `${text} Proposal ready (fit IoU ${p.fit_iou.toFixed(3)})${p.stale ? ", for an earlier mask" : ""}.`;
    if (p?.status === "no_trace") return `${text} The model found no body to propose.`;
    if (p?.status === "error") return `${text} Proposal failed: ${p.error}`;
    if (!this.autoPropose) return `${text} Propose computes the model's body (slow without a GPU).`;
    return text;
  }

  // ------------------------------------------------------------------ layers

  toggleLayer(id) {
    if (id === "mask") { this.toggleMask(); return; }
    this.view.setLayerVisible(id, !this.view.isLayerVisible(id));
    this._syncLayers();
  }

  toggleMask() {
    this.maskShown = !this.maskShown;
    this._syncLayers();
    this.view.redraw();
  }

  _syncLayers() {
    this.view.setLayerVisible("mask", this.maskShown);
    for (const [id, b] of Object.entries(this.layerButtons)) b.setAttribute("aria-pressed", String(this.view.isLayerVisible(id)));
  }

  toggleRaw() {
    this.raw = !this.raw;
    this.rawButton.setAttribute("aria-pressed", String(this.raw));
    this._renderBase();
  }

  _renderMask() {
    this.maskOverlay = maskOverlay(this.mask, this.width, this.height);
    this.previewOverlay = this.previewMask ? maskOverlay(this.previewMask, this.width, this.height, true) : null;
    this.view.redraw();
  }

  _apCanvas() {
    const values = this.body?.fit?.apValues || (this.proposal?.status === "ready" ? this.proposal.apValues : null);
    if (!values) return null;
    if (this._apSource !== values) { this._apSource = values; this._apOverlay = apOverlay(values, this.width, this.height); }
    return this._apOverlay;
  }

  // ------------------------------------------------------------------ painting

  setBrush(value) { this.brush = value; this.cancelPreview(); this._syncControls(); }

  _pointer({type, x, y}) {
    if (!this.frame) return;
    if (this.trace) {
      if (type === "down" && !this.trace.fit) { this.trace.points.push([x, y]); this._syncControls(); this.view.redraw(); }
      return;
    }
    this.cursor = {x, y};
    if (type === "down") {
      this.cancelPreview();
      this._remember();
      this.painting = {x, y};
      stroke(this.mask, this.width, this.height, this.painting, {x, y}, this.size, this.brush);
      this._renderMask();
    } else if (type === "move" && this.painting) {
      stroke(this.mask, this.width, this.height, this.painting, {x, y}, this.size, this.brush);
      this.painting = {x, y};
      this._renderMask();
    } else if (type === "up" && this.painting) {
      this.painting = null;
      this._maskEdited();
    } else {
      this.view.redraw();
    }
  }

  _remember() {
    this.history.push(this.mask.slice());
    if (this.history.length > UNDO_LIMIT) this.history.shift();
  }

  _maskEdited() {
    if (this.proposal?.status === "ready") this.proposal.stale = true;
    this._changed();
  }

  undo() {
    if (!this.history.length) return;
    this.mask = this.history.pop();
    this._renderMask();
    this._maskEdited();
  }

  revert() {
    if (sameMask(this.mask, this.openedMask)) return;
    this._remember();
    this.mask = this.openedMask.slice();
    this.cancelPreview();
    this._renderMask();
    this._maskEdited();
  }

  // ------------------------------------------------------------------ proposals

  async preview(source) {
    if (!this.frame || this.trace) return;
    if (source === "network" && !this.probability) {
      if (!this.frame.models.mask) { this.ctx.toast("This setup has no mask model yet; try Threshold.", "error"); return; }
      this.busy++; this._status("Running the model…"); this._syncControls();
      try {
        const answer = await post("/api/labeling/network", this.request);
        this.probability = await grayOf(answer.probability);
        this._status("");
      } catch (error) {
        this._status(error.message, "error");
        return;
      } finally { this.busy--; this._syncControls(); }
    }
    this.previewSource = source;
    this._updatePreview();
  }

  _updatePreview() {
    if (!this.previewSource) return;
    this.previewMask = this.previewSource === "network" ? thresholded(this.probability, this.threshold) : thresholded(this.gray, this.threshold, true);
    this._renderMask();
    this._syncControls();
  }

  setThreshold(value) {
    this.threshold = value;
    this.thresholdValue.textContent = value.toFixed(2);
    this._updatePreview();
  }

  applyPreview() {
    if (!this.previewMask) return;
    this._remember();
    this.mask = applyProposal(this.mask, this.previewMask);
    this.previewMask = null;
    this.previewSource = null;
    this._renderMask();
    this._maskEdited();
  }

  cancelPreview() {
    if (!this.previewMask) return;
    this.previewMask = null;
    this.previewSource = null;
    this._renderMask();
    this._syncControls();
  }

  async refine(method) {
    if (!this.frame || this.busy) return;
    this.cancelPreview();
    this.busy++; this._syncControls();
    const generation = this.generation;
    try {
      const answer = await post("/api/labeling/refine", {mask: encodeMask(this.mask, this.width, this.height), width: this.width, height: this.height, method});
      const refined = await decodeMask(answer.mask);
      if (generation !== this.generation) return;
      this._remember();
      this.mask = refined;
      this._renderMask();
      this._maskEdited();
    } catch (error) {
      this._status(error.message, "error");
    } finally { this.busy--; this._syncControls(); }
  }

  // ------------------------------------------------------------------ body

  // The proposal opens with the frame only when the server's models run on a GPU.
  get autoPropose() { return !!this.ctx.config?.gpu; }

  async requestProposal() {
    if (!this.frame?.models.body) return null;
    const generation = this.generation, mask = this.mask.slice();
    this.proposal = {status: "loading"};
    this._syncControls();
    try {
      const answer = await post("/api/labeling/proposal", {...this.request, mask: encodeMask(mask, this.width, this.height)});
      if (generation !== this.generation) return null;
      if (answer.status === "ready") answer.apValues = await grayOf(answer.ap);
      answer.stale = !sameMask(mask, this.mask);
      this.proposal = answer;
    } catch (error) {
      if (generation !== this.generation) return null;
      this.proposal = {status: "error", error: error.message};
    }
    this._syncControls();
    this.view.redraw();
    return this.proposal;
  }

  async useProposal() {
    if (!this.frame || this.trace) return;
    let p = this.proposal;
    if (p?.status === "loading") { this._status("The proposal is still being computed…"); return; }
    if (!p || p.stale || p.status === "error") p = await this.requestProposal();
    if (p?.status !== "ready") { this.ctx.toast("There is no proposal for this frame; trace the midline instead.", "error"); return; }
    this.body = {kind: "trace", trace: p.trace_xy, head: null, fit: p};
    this.bodyChanged = true;
    this._changed();
    this.view.redraw();
  }

  flip() {
    if (!this.frame || this.trace) return;
    const body = this.body;
    const fit = body.fit ? reversedFit(body.fit) : null;
    if (body.kind === "trace") {
      this.body = {kind: "trace", trace: [...body.trace].reverse(), head: null, fit};
    } else if (fit && fit.head_xy) {
      this.body = {kind: "head", trace: null, head: fit.head_xy, fit};
    } else if (this.proposal?.status === "ready") {
      const reversed = reversedFit(this.proposal);
      this.body = {kind: "trace", trace: [...this.proposal.trace_xy].reverse(), head: null, fit: reversed};
    } else {
      this.ctx.toast("Nothing to flip yet: use the proposal or trace the midline.", "error");
      return;
    }
    this.bodyChanged = true;
    this._changed();
    this.view.redraw();
  }

  startTrace(fromProposal) {
    if (!this.frame || this.trace) return;
    if (fromProposal && this.proposal?.status !== "ready") return;
    this.cancelPreview();
    this.trace = {points: fromProposal ? this.proposal.trace_xy.map((p) => [...p]) : [], fit: null};
    this.view.setLayerVisible("midline", false);
    this._syncLayers();
    this._syncControls();
    this.view.redraw();
  }

  removeLastPoint() {
    if (!this.trace || this.trace.fit) return;
    this.trace.points.pop();
    this._syncControls();
    this.view.redraw();
  }

  async fitTrace() {
    if (!this.trace || this.trace.points.length < 2 || this.busy) return;
    this.busy++; this._syncControls();
    this.traceNote.textContent = "Fitting the body along the trace…";
    const generation = this.generation;
    try {
      const fit = await post("/api/labeling/fit", {...this.request, mask: encodeMask(this.mask, this.width, this.height), trace_xy: this.trace.points});
      if (generation !== this.generation || !this.trace) return;
      fit.apValues = await grayOf(fit.ap);
      this.trace.fit = fit;
    } catch (error) {
      this._status(error.message, "error");
    } finally { this.busy--; this._syncControls(); this.view.redraw(); }
  }

  acceptTrace() {
    if (!this.trace?.fit) return;
    this.body = {kind: "trace", trace: this.trace.points, head: null, fit: this.trace.fit};
    this.bodyChanged = true;
    this.endTrace();
  }

  endTrace() {
    if (!this.trace) return;
    this.trace = null;
    this.view.setLayerVisible("midline", true);
    this._syncLayers();
    this._changed();
    this.view.redraw();
  }

  // ------------------------------------------------------------------ context

  async _ensureContext() {
    if (this.context || this.contextLoading) return !!this.context;
    const generation = this.generation;
    this.contextLoading = true;
    this.contextNote.textContent = "Loading the context frames…";
    try {
      const answer = await post("/api/labeling/context", this.request);
      const frames = await Promise.all(answer.frames.map(grayOf));
      if (generation !== this.generation) return false;
      this.context = {valid: answer.valid, frames, maxLag: answer.max_lag};
      return true;
    } catch (error) {
      this.contextNote.textContent = error.message;
      return false;
    } finally { this.contextLoading = false; this._syncContext(); }
  }

  async setContextMode(mode) {
    if (!this.frame) return;
    this.contextMode = mode;
    if (mode !== "frame") this.stopPlay();
    if ((mode === "difference" || this.offset) && !(await this._ensureContext())) return;
    this._renderBase();
    this._syncContext();
  }

  async setOffset(value) {
    if (!this.frame) return;
    const max = this.frame.max_lag;
    this.offset = Math.max(-max, Math.min(max, value));
    if (this.offset && !(await this._ensureContext())) return;
    this._renderBase();
    this._syncContext();
  }

  async setLag(value) {
    this.lag = Math.max(1, Math.min(this.frame?.max_lag || 16, value));
    if (this.contextMode === "difference") this._renderBase();
    this._syncContext();
  }

  async togglePlay() {
    if (this.playTimer) { this.stopPlay(); return; }
    if (!this.frame || !(await this._ensureContext())) return;
    if (this.contextMode !== "frame") await this.setContextMode("frame");
    const max = this.frame.max_lag;
    this.playTimer = setInterval(() => this.setOffset(this.offset >= max ? -max : this.offset + 1), 1000 / PLAY_FPS);
    this._syncContext();
  }

  stopPlay() {
    if (!this.playTimer) return;
    clearInterval(this.playTimer);
    this.playTimer = null;
    this._syncContext();
  }

  _renderBase() {
    if (!this.frame) return;
    const L = this.frame.max_lag, c = this.context;
    let base;
    if (this.contextMode === "difference" && c) base = differenceCanvas(c.frames[L + this.lag], c.frames[L - this.lag], this.width, this.height);
    else if (this.offset && c) base = grayCanvas(c.frames[L + this.offset], this.width, this.height);
    else base = this.raw ? this.images.raw : this.images.flat;
    this.view.setImage(base);
  }

  _syncContext() {
    if (!this.frame) return;
    const difference = this.contextMode === "difference";
    this.modeButtons.frame.setAttribute("aria-pressed", String(!difference));
    this.modeButtons.difference.setAttribute("aria-pressed", String(difference));
    this.offsetGroup.hidden = difference;
    this.lagGroup.hidden = !difference;
    this.offsetInput.value = this.offset;
    this.lagInput.value = this.lag;
    this.offsetValue.textContent = this.offset ? `t${this.offset > 0 ? "+" : ""}${this.offset}` : "t";
    this.lagValue.textContent = `±${this.lag}`;
    this.playButton.textContent = this.playTimer ? "Pause" : "Play";
    if (this.contextLoading) return;
    const valid = this.frame.context_valid, L = this.frame.max_lag;
    let note = "";
    if (difference) note = valid[L + this.lag] && valid[L - this.lag] ? "" : "one end is outside the recording";
    else if (this.offset && !valid[L + this.offset]) note = "outside the recording: the nearest frame repeats";
    this.contextNote.textContent = note;
  }

  // ------------------------------------------------------------------ saving

  savePayload() {
    const body = this.body, payload = {
      ...this.request, mask: encodeMask(this.mask, this.width, this.height), mask_only: this.maskOnly.checked,
      expected_revision: this.frame.expected_revision,
    };
    if (body.kind === "trace") payload.trace_xy = body.trace;
    else if (body.kind === "head") payload.head_xy = body.head;
    else if (body.fit?.head_xy) payload.head_xy = body.fit.head_xy;  // the shown body: saving accepts it
    return payload;
  }

  async save({next}) {
    if (!this.frame || this.busy || this.trace) return null;
    if (this.previewMask) this.applyPreview();
    this.busy++; this._syncControls();
    this._status("Saving…");
    try {
      const answer = await post("/api/labeling/save", this.savePayload());
      this._status(`Saved revision ${answer.label.revision} to ${answer.dataset}${answer.job ? "; body targets are rebuilding" : ""}.`, "ok");
      this.frame.label = answer.label;
      this.frame.expected_revision = answer.label.revision;
      this.frame.body = {...this.frame.body, mask_only: this.maskOnly.checked};
      this.openedMask = this.mask.slice();
      this.bodyChanged = false;
      this.onSaved?.(answer, {next});
      return answer;
    } catch (error) {
      this._status(error.message, "error");
      this.ctx.toast(`Could not save: ${error.message}`, "error");
      return null;
    } finally { this.busy--; this._changed(); }
  }

  // ------------------------------------------------------------------ keys

  // Returns true when the key was the editor's.
  handleKey(event) {
    if (!this.frame) return false;
    const key = event.key, lower = key.length === 1 ? key.toLowerCase() : key;
    if (event.ctrlKey || event.metaKey) {
      if (lower === "z") { this.undo(); return true; }
      return false;
    }
    if (this.trace) {
      if (key === "Enter") { this.trace.fit ? this.acceptTrace() : this.fitTrace(); return true; }
      if (key === "Backspace") { this.removeLastPoint(); return true; }
      if (key === "Escape") { this.endTrace(); return true; }
    }
    if (key === "Escape" && this.previewMask) { this.cancelPreview(); return true; }
    const actions = {
      w: () => this.setBrush(WORM), e: () => this.setBrush(BACKGROUND),
      "[": () => this._resize(-2), "]": () => this._resize(2),
      n: () => this.preview("network"), a: () => this.applyPreview(), z: () => this.undo(),
      m: () => this.toggleMask(), r: () => this.toggleRaw(), f: () => this.view.fit(),
      g: () => this.useProposal(), h: () => this.flip(), t: () => this.startTrace(false),
      d: () => this.setContextMode(this.contextMode === "frame" ? "difference" : "frame"),
      " ": () => this.togglePlay(), s: () => this.save({next: false}), Enter: () => this.save({next: true}),
      ArrowLeft: () => (this.contextMode === "frame" ? this.setOffset(this.offset - 1) : this.setLag(this.lag - 1)),
      ArrowRight: () => (this.contextMode === "frame" ? this.setOffset(this.offset + 1) : this.setLag(this.lag + 1)),
    };
    const action = actions[lower] || actions[key];
    if (!action || event.shiftKey || event.altKey) return false;
    action();
    return true;
  }

  _resize(step) {
    this.size = Math.max(2, Math.min(60, this.size + step));
    this.sizeInput.value = this.size;
    this.sizeValue.textContent = `${this.size}px`;
    this.view.redraw();
  }
}

export const SHORTCUTS = [
  ["W / E", "Worm / Background brush"], ["[ ]", "Brush size"], ["N", "Network proposal"], ["A", "Apply the proposal"],
  ["Z", "Undo"], ["G", "Use the body proposal"], ["H", "Flip head/tail"], ["T", "Trace the midline (Enter fits, Backspace removes a point, Esc cancels)"],
  ["D", "Frames / Difference"], ["Space", "Play the context"], ["← →", "Context offset (or lag)"], ["M", "Mask on/off"],
  ["R", "Raw / flat-fielded frame"], ["F", "Fit the view (or double-click)"], ["Enter", "Save & next"], ["S", "Save"],
  ["Shift + ← →", "Previous / next frame"], ["Shift + drag, right drag", "Pan; wheel zooms"],
];

// ------------------------------------------------------------------ drawing helpers

function reversedFit(fit) {
  return {
    ...fit, centerline_xy: [...fit.centerline_xy].reverse(), width_profile: fit.width_profile ? [...fit.width_profile].reverse() : fit.width_profile,
    head_xy: fit.tail_xy, tail_xy: fit.head_xy, apValues: fit.apValues ? reversedAp(fit.apValues) : null,
    trace_xy: fit.trace_xy ? [...fit.trace_xy].reverse() : fit.trace_xy,
  };
}

function drawPoints(g, view, points, color) {
  if (!points?.length) return;
  const px = 1 / view.scale;
  g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 1.5 * px;
  g.beginPath();
  points.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.stroke();
  points.forEach(([x, y], i) => {
    g.beginPath();
    g.arc(x, y, (i === 0 ? 5 : 3.5) * px, 0, Math.PI * 2);
    g.fill();
  });
}

function drawNose(g, view, [x, y]) {
  const r = 6 / view.scale;
  g.strokeStyle = "#ffffff"; g.lineWidth = 1.5 / view.scale;
  g.beginPath();
  g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath();
  g.stroke();
}

function drawTube(g, view, centerline, widths) {
  if (!widths) return;
  const left = [], right = [];
  for (let i = 0; i < centerline.length; i++) {
    const a = centerline[Math.max(0, i - 1)], b = centerline[Math.min(centerline.length - 1, i + 1)];
    const dx = b[0] - a[0], dy = b[1] - a[1], n = Math.hypot(dx, dy) || 1, r = widths[i] / 2;
    left.push([centerline[i][0] - dy / n * r, centerline[i][1] + dx / n * r]);
    right.push([centerline[i][0] + dy / n * r, centerline[i][1] - dx / n * r]);
  }
  g.strokeStyle = "#e6ebef"; g.lineWidth = 1 / view.scale;
  g.beginPath();
  [...left, ...right.reverse()].forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.closePath();
  g.stroke();
}

async function overlapCanvasOf(url, width, height) {
  const values = await grayOf(url);
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const g = canvas.getContext("2d"), image = g.createImageData(width, height);
  for (let i = 0; i < values.length; i++) if (values[i]) { const o = i * 4; image.data[o] = 255; image.data[o + 1] = 255; image.data[o + 3] = 255; }
  g.putImageData(image, 0, 0);
  return canvas;
}

