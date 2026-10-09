// The Workspace's developer tools, present only when the app runs with
// --dev (docs/APP_SIMPLIFICATION.md, section 2): the raw (not flat-fielded)
// frame, every layer the frame payload carries, the "Frame details" drawer
// (classification, ambiguity flags, width and curvature along the body, mask
// pipeline statistics, the workspace summary), one extra per-frame series on
// the timeline, the jobs drawer, and the algorithm override for Refit.

import {api, el, post, query} from "./api.js";
import {decodeGray, loadImage} from "./workspace_draw.js";
import {maskCanvas, drawMidline} from "./frame_canvas.js";

// Raster layers of the frame payload, drawn with these colours.
const RASTERS = [
  ["probability", "Network probability", null],
  ["mask_raw", "Mask: thresholded", [255, 209, 102]],
  ["mask_filled", "Mask: holes filled", [255, 140, 66]],
  ["mask_largest", "Mask: largest part", [184, 120, 255]],
  ["mask_override", "Mask: edited", [255, 79, 163]],
  ["tube", "Fitted tube", [87, 214, 141]],
  ["tube_independent", "Independent fit tube", [108, 180, 255]],
];
const SERIES = ["iou", "ambiguity_score", "body_length_px", "width_px", "energy", "points_in_fov", "taper_asymmetry", "orientation_gap",
  "self_contact_px", "pose_jump_px", "length_deviation", "area_ratio", "tube_coverage"];

export class DevTools {
  constructor(view) {
    this.view = view;
    this.raw = false;
    this.on = new Set();
    this.rasters = new Map();
    this.payload = null;
    this.overlap = null;
    this.details = el("aside", {class: "ws-drawer", hidden: true});
    view.center.append(this.details);
    this.rawButton = el("button", {type: "button", class: "dev-only", "aria-pressed": "false", title: "Show the frame before flat-fielding", onclick: () => {
      this.raw = !this.raw; this.rawButton.setAttribute("aria-pressed", String(this.raw)); view.seek(view.row);
    }}, "Raw");
    this.layerMenu = el("details", {class: "ws-menu dev-only"}, el("summary", {}, "Layers"), el("div", {class: "ws-menu-body"}));
    this.detailsButton = el("button", {type: "button", class: "dev-only", "aria-pressed": "false", onclick: () => this.toggleDetails()}, "Details");
    view.toolbar.append(this.rawButton, this.layerMenu, this.detailsButton);
    this.seriesSelect = el("select", {class: "dev-only", "aria-label": "Extra series", onchange: () => this.showSeries()}, el("option", {value: ""}, "No extra series"));
    view.transport.insertBefore(this.seriesSelect, view.transport.querySelector(".ws-timeline-hint"));
    this.renderLayerMenu();
  }

  reset() {
    this.payload = null;
    this.rasters.clear();
    this.overlap = null;
    this.seriesSelect.value = "";
    this.view.timeline.set("series", null);
  }

  // The full workspace payload: its per-frame series and summary.
  async load() {
    try { this.payload = await api(this.view.base); } catch { this.payload = null; return; }
    const series = this.payload.series || {};
    const current = this.seriesSelect.value;
    this.seriesSelect.replaceChildren(el("option", {value: ""}, "No extra series"),
      ...SERIES.filter((key) => Array.isArray(series[key])).map((key) => el("option", {value: key, selected: key === current}, key)),
      ...Object.keys(series.flags || {}).map((flag) => el("option", {value: `flag:${flag}`, selected: `flag:${flag}` === current}, `flag ${flag}`)));
    this.showSeries();
    this.renderDetails();
  }

  showSeries() {
    const key = this.seriesSelect.value, series = this.payload?.series || {};
    const values = !key ? null : key.startsWith("flag:") ? series.flags?.[key.slice(5)] : series[key];
    this.view.timeline.set("series", values ? {values: values.map((v) => (v === null ? NaN : Number(v))), label: key} : null);
  }

  renderLayerMenu() {
    const body = this.layerMenu.querySelector(".ws-menu-body");
    const box = (id, label) => el("label", {class: "inline"}, el("input", {type: "checkbox", checked: this.on.has(id), onchange: (event) => {
      if (event.target.checked) this.on.add(id); else this.on.delete(id);
      this.view.canvas.redraw();
      if (event.target.checked && id === "overlap") this.view.loadFields(this.view.frameOf(this.view.row), this.view.generation);
    }}), label);
    body.replaceChildren(...RASTERS.map(([id, label]) => box(id, label)), box("independent", "Independent fit midline"), box("overlap", "Network crossings"));
  }

  async frameLoaded(payload) {
    this.rasters.clear();
    const frame = payload.frame_index;
    await Promise.all(RASTERS.map(async ([id, , color]) => {
      const url = payload.layers?.[id];
      if (!url) return;
      if (!color) { this.rasters.set(id, await loadImage(url)); return; }
      const gray = await decodeGray(url);
      if (gray && payload.frame_index === frame) this.rasters.set(id, maskCanvas(gray.data, gray.width, gray.height, color, 0.4));
    }));
    if (this.view.current === payload) this.view.canvas.redraw();
    this.renderDetails();
  }

  async fieldsLoaded(frame, fields) {
    const gray = await decodeGray(fields.overlap);
    this.overlap = {frame, canvas: gray ? maskCanvas(gray.data, gray.width, gray.height, [255, 93, 93], 0.6) : null, head: fields.head_xy, tail: fields.tail_xy};
    this.view.canvas.redraw();
  }

  drawLayers(g, v) {
    for (const [id] of RASTERS) {
      const image = this.rasters.get(id);
      if (!this.on.has(id) || !image) continue;
      if (id === "probability") { g.globalAlpha = 0.5; g.drawImage(image, 0, 0); g.globalAlpha = 1; } else g.drawImage(image, 0, 0);
    }
    const pose = this.view.current?.pose;
    if (this.on.has("independent") && pose?.independent) drawMidline(g, v, pose.independent.centerline_xy, {color: "#6cb4ff", head: "#6cb4ff", tail: "#6cb4ff", width: 1.5});
    if (this.on.has("overlap") && this.overlap?.canvas && this.overlap.frame === this.view.frameOf(this.view.row)) g.drawImage(this.overlap.canvas, 0, 0);
  }

  toggleDetails() {
    this.details.hidden = !this.details.hidden;
    this.detailsButton.setAttribute("aria-pressed", String(!this.details.hidden));
    this.renderDetails();
  }

  renderDetails() {
    if (this.details.hidden) return;
    const payload = this.view.current, stats = payload?.stats || {}, pose = payload?.pose;
    const flags = (stats.flags || []).map((f) => el("tr", {class: f.fired ? "fired" : ""}, el("td", {}, f.name), el("td", {class: "num"}, String(f.value ?? "—")),
      el("td", {class: "note"}, f.threshold === null || f.threshold === undefined ? "" : `${f.test} ${f.threshold}`)));
    const maskStats = {...(payload?.mask_stats || {}), ...Object.fromEntries(Object.entries(payload?.mask_stats_stored || {}).map(([k, v]) => [`stored ${k}`, v]))};
    const summary = this.payload?.summary || {};
    this.details.replaceChildren(
      el("div", {class: "ws-panel-head"}, el("h3", {}, `Frame ${payload?.frame_index ?? ""}`), el("button", {class: "link", onclick: () => this.toggleDetails()}, "Close")),
      el("p", {class: "note"}, `classification: ${stats.classification?.kind || "—"} · source ${stats.source_name || "—"} · iou ${stats.iou ?? "—"} · score ${stats.ambiguity_score ?? "—"}`),
      el("h3", {}, "Ambiguity flags"), el("table", {class: "data ws-flags"}, el("tbody", {}, flags)),
      el("h3", {}, "Width along the body"), sparkline(pose?.width_profile, "#ffd166"),
      el("h3", {}, "Curvature along the body"), sparkline(pose?.curvature, "#6cb4ff", true),
      el("h3", {}, "Mask pipeline"), el("table", {class: "data"}, el("tbody", {}, Object.entries(maskStats).map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {class: "num"}, String(v)))))),
      el("h3", {}, "Workspace"), el("table", {class: "data"}, el("tbody", {}, Object.entries(summary).filter(([, v]) => typeof v !== "object").map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {class: "num"}, String(v)))))),
    );
  }

  // Refit's algorithm override.
  refitOverride() {
    if (!this.algorithms) {
      this.algorithms = [];
      api("/api/algorithms").then((list) => { this.algorithms = list; this.view.renderFix(); }).catch(() => {});
    }
    this.algorithm ??= "";
    this.params ??= "";
    return el("div", {class: "ws-dev-refit dev-only"},
      el("label", {class: "inline"}, "Refit with", el("select", {onchange: (event) => { this.algorithm = event.target.value; }},
        el("option", {value: ""}, "recommended"), this.algorithms.map((a) => el("option", {value: a.id, selected: a.id === this.algorithm}, a.id)))),
      el("input", {type: "text", placeholder: "params JSON", value: this.params, "aria-label": "Refit parameters", onchange: (event) => { this.params = event.target.value; }}));
  }

  refitParams() {
    const out = {};
    if (this.algorithm) out.algorithm = this.algorithm;
    if (this.params?.trim()) {
      try { out.params = JSON.parse(this.params); } catch { throw new Error("The refit parameters are not valid JSON"); }
    }
    return out;
  }

  // The jobs drawer: this workspace's jobs with progress, Cancel and the log.
  openJobs() {
    const dialog = el("dialog", {class: "ws-dialog ws-jobs"}), body = el("div", {});
    let timer = null;
    const render = async () => {
      let jobs = [];
      try { jobs = (await api("/api/jobs")).filter((job) => job.spec.workspace === this.view.name); } catch (error) { body.replaceChildren(el("p", {class: "note error"}, error.message)); return; }
      body.replaceChildren(jobs.length ? el("table", {class: "data"}, el("tbody", {}, jobs.slice(0, 30).map((job) => el("tr", {},
        el("td", {}, job.id), el("td", {}, job.spec.label || job.spec.kind), el("td", {}, job.state + (job.slurm_state ? ` (${job.slurm_state})` : "")),
        el("td", {class: "num"}, `${Math.round(job.progress * 100)}%`), el("td", {class: "note ws-job-message"}, job.error ? job.error.split("\n").filter(Boolean).pop() : job.message),
        el("td", {}, ["queued", "running"].includes(job.state) ? el("button", {onclick: () => post(`/api/jobs/${job.id}/cancel`).then(render)}, "Cancel") : null,
          el("button", {class: "link", onclick: async () => {
            const log = await api(`/api/jobs/${job.id}/log${query({tail: 60})}`);
            body.append(el("pre", {class: "ws-log"}, log.log || "(empty log)"));
          }}, "Log")))))) : el("p", {class: "note"}, "No jobs on this workspace."));
      timer = setTimeout(render, 2000);
    };
    dialog.append(el("h2", {}, `Jobs · ${this.view.name}`), body, el("div", {class: "ws-dialog-actions"}, el("button", {onclick: () => dialog.close()}, "Close")));
    dialog.addEventListener("close", () => { clearTimeout(timer); dialog.remove(); });
    document.body.append(dialog);
    dialog.showModal();
    render();
  }
}

function sparkline(values, color, signed = false) {
  if (!values?.length) return el("p", {class: "note"}, "no pose");
  const w = 240, h = 48, lo = signed ? -Math.max(...values.map(Math.abs)) : Math.min(0, ...values), hi = signed ? -lo : Math.max(...values);
  const span = hi - lo || 1;
  const points = values.map((v, i) => `${(i / (values.length - 1) * w).toFixed(1)},${(h - (v - lo) / span * h).toFixed(1)}`).join(" ");
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("class", "ws-spark");
  if (signed) { const zero = document.createElementNS(ns, "line"); zero.setAttribute("x1", 0); zero.setAttribute("x2", w); zero.setAttribute("y1", h / 2); zero.setAttribute("y2", h / 2); zero.setAttribute("stroke", "#2a343c"); svg.append(zero); }
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", points); line.setAttribute("fill", "none"); line.setAttribute("stroke", color); line.setAttribute("stroke-width", "1.5");
  svg.append(line);
  return el("div", {}, svg, el("span", {class: "note"}, `${(+lo.toFixed(3))} … ${(+hi.toFixed(3))}`));
}
