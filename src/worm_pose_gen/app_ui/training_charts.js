// Loss curves for the Training page: train and validation loss per epoch as
// one small SVG, used live in a running job's card and in a model's Details.
// curve: [{epoch, train_loss, val_loss}] (null where an epoch has no value).

const NS = "http://www.w3.org/2000/svg";

function node(tag, attributes = {}, text = null) {
  const element = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text !== null) element.textContent = text;
  return element;
}

export function lossChart(curve, {width = 360, height = 140, maxEpochs = null} = {}) {
  const svg = node("svg", {class: "tr-chart", viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
    "aria-label": "Training and validation loss by epoch"});
  const pad = {left: 40, right: 8, top: 8, bottom: 20};
  const values = curve.flatMap((c) => [c.train_loss, c.val_loss]).filter((v) => v !== null && v !== undefined && isFinite(v));
  if (!values.length) {
    svg.append(node("text", {x: width / 2, y: height / 2, "text-anchor": "middle", class: "tr-chart-empty"}, "no epoch finished yet"));
    return svg;
  }
  let low = Math.min(...values), high = Math.max(...values);
  if (high - low < 1e-9) { low -= 0.5 * Math.abs(low || 1); high += 0.5 * Math.abs(high || 1); }
  const last = Math.max(maxEpochs || 0, ...curve.map((c) => c.epoch), 2);
  const x = (epoch) => pad.left + ((epoch - 1) / (last - 1)) * (width - pad.left - pad.right);
  const y = (value) => pad.top + (1 - (value - low) / (high - low)) * (height - pad.top - pad.bottom);
  svg.append(
    node("line", {x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom, class: "tr-axis"}),
    node("line", {x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom, class: "tr-axis"}),
    node("text", {x: pad.left - 4, y: pad.top + 8, "text-anchor": "end", class: "tr-tick"}, high.toPrecision(2)),
    node("text", {x: pad.left - 4, y: height - pad.bottom, "text-anchor": "end", class: "tr-tick"}, low.toPrecision(2)),
    node("text", {x: pad.left, y: height - 4, class: "tr-tick"}, "epoch 1"),
    node("text", {x: width - pad.right, y: height - 4, "text-anchor": "end", class: "tr-tick"}, String(last)),
  );
  for (const [key, cls] of [["train_loss", "tr-train"], ["val_loss", "tr-val"]]) {
    const points = curve.filter((c) => c[key] !== null && c[key] !== undefined).map((c) => [x(c.epoch), y(c[key])]);
    if (points.length > 1) svg.append(node("polyline", {points: points.map((p) => p.join(",")).join(" "), class: cls}));
    for (const [px, py] of points.length === 1 ? points : []) svg.append(node("circle", {cx: px, cy: py, r: 2.5, class: cls}));
  }
  const best = curve.filter((c) => c.val_loss !== null && c.val_loss !== undefined).reduce((a, c) => (!a || c.val_loss < a.val_loss ? c : a), null);
  if (best) svg.append(node("circle", {cx: x(best.epoch), cy: y(best.val_loss), r: 3.5, class: "tr-best"}));
  return svg;
}

export function chartLegend() {
  const span = document.createElement("span");
  span.className = "tr-legend";
  span.innerHTML = '<i class="tr-train"></i>train <i class="tr-val"></i>validation <i class="tr-best"></i>kept';
  return span;
}
