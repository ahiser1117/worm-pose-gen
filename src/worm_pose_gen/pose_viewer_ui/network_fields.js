"use strict";

// The body-field network's prediction on the shown workspace frame
// (GET /api/workspaces/<name>/network-fields, app/network_fields.py), drawn by
// the "Network A-P field" and "Network head / tail" layers. Nothing is
// fetched while both are off. One request is in flight at a time and asks for
// the frame shown when it starts; a response for a frame no longer shown is
// dropped and the latest frame is asked for next, so playback waits on one
// prediction at most. The last prediction stays drawn until the next arrives.
window.networkFields = (() => {
  let shown = null, pending = null, failure = null, generation = 0;
  const overlay = document.createElement("canvas");
  const active = id => { const l = layer(id); return l && l.on && l.alpha > 0 ? l : null; };
  const wanted = () => isWorkspace() && serverIsApp() && !!state.run && (layer("net_ap").on || layer("net_ends").on);
  const keyOf = () => `${state.runName}:${currentFrameIndex()}`;
  const current = () => shown && shown.workspace === state.runName ? shown : null;
  function status(message, kind) {
    const node = $("#network-fields-status"); if (!node) return;
    node.textContent = message; node.className = "note" + (kind ? " " + kind : "");
  }
  function endText(name, xy, peak, threshold) {
    if (xy) return `${name} peak ${peak.toFixed(2)}`;
    return peak < threshold ? `${name} not in view (peak ${peak.toFixed(2)})` : `${name} dropped: at the image edge or off the mask (peak ${peak.toFixed(2)})`;
  }
  function describe() {
    const p = current();
    if (!wanted()) status("");
    else if (failure && (failure.unconfigured || failure.key === keyOf())) status(`Network layers: ${failure.message}`, "error");
    else if (p) status(`Network · frame ${p.frame}${p.key === keyOf() ? "" : " (updating)"} · ${endText("head", p.head_xy, p.head_peak, p.end_threshold)} · ${endText("tail", p.tail_xy, p.tail_peak, p.end_threshold)}`);
    else status("Network · predicting…");
  }
  async function install(payload, key) {
    const [ap, crossing] = await Promise.all([decodeGray(payload.ap), decodeGray(payload.overlap)]);
    const {width, height} = payload;
    overlay.width = width; overlay.height = height;
    const c = overlay.getContext("2d"), image = c.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const rgb = ap[i] ? viridis((ap[i] - 1) / 254) : crossing[i] ? [255, 255, 255] : null;
      if (rgb) image.data.set([...rgb, 255], i * 4);
    }
    c.putImageData(image, 0, 0);
    shown = {...payload, key, workspace: state.runName, _ap: ap, _crossing: crossing};
  }
  async function request() {
    if (pending || !wanted()) return;
    const key = keyOf(), token = generation;
    if (current()?.key === key || failure?.key === key) return;
    pending = key; describe();
    try {
      const payload = await api(sourceUrl("workspace", state.runName, "network-fields", `frame=${currentFrameIndex()}`));
      if (token !== generation || key !== keyOf()) return;  // stale: the frame or source changed meanwhile
      await install(payload, key);
      failure = null;
    } catch (error) {
      if (token === generation) failure = {key, message: error.message, unconfigured: /--body-net/.test(error.message)};
    } finally {
      if (token === generation) {
        pending = null; describe(); renderLegend(); renderLayerAvailability(); draw();
        if (!failure?.unconfigured) request();
      }
    }
  }
  // A new frame is shown (any detail, so playback too).
  function onFrame() { if (wanted()) request(); else describe(); }
  function onLayer() { if (failure?.unconfigured) failure = null; onFrame(); renderLayerAvailability(); }
  function onSource() { generation++; shown = null; pending = null; failure = null; describe(); }
  // Under the poses: the A-P field (crossings white).
  function drawField(g) {
    const p = current(), l = active("net_ap");
    if (!p || !l) return;
    g.save(); g.globalAlpha = l.alpha; g.imageSmoothingEnabled = false; g.drawImage(overlay, 0, 0); g.restore();
  }
  // Over the poses: filled discs with a white rim and a letter, unlike the fitted pose's square head and ring tail.
  function drawEnds(g) {
    const p = current(), l = active("net_ends");
    if (!p || !l) return;
    const s = state.view.scale;
    g.save(); g.globalAlpha = l.alpha; g.font = `bold ${9 / s}px system-ui`; g.textAlign = "center"; g.textBaseline = "middle";
    for (const [xy, colour, letter] of [[p.head_xy, "rgb(60,255,60)", "H"], [p.tail_xy, "rgb(255,70,70)", "T"]]) {
      if (!xy) continue;  // not in view, or dropped as the fitter drops it
      g.beginPath(); g.arc(xy[0], xy[1], 6 / s, 0, Math.PI * 2); g.fillStyle = colour; g.fill();
      g.lineWidth = 1.5 / s; g.strokeStyle = "#fff"; g.stroke();
      g.fillStyle = "#000"; g.fillText(letter, xy[0], xy[1] + 0.5 / s);
    }
    g.restore();
  }
  function readout(x, y) {
    const p = current();
    if (!p || !active("net_ap") || x >= p.width || y >= p.height) return "";
    const i = y * p.width + x;
    return p._ap[i] ? `network A-P ${((p._ap[i] - 1) / 254).toFixed(2)}` : p._crossing[i] ? "network crossing" : "";
  }
  return {
    onFrame, onLayer, onSource, drawField, drawEnds, readout,
    available: () => !!current(),
    usable: () => isWorkspace() && !failure?.unconfigured,
    shown: () => current(),
  };
})();
window.addEventListener("workflow:source", () => networkFields.onSource());
