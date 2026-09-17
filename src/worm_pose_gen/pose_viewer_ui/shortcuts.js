"use strict";

// A chord has one meaning in every task. Disabled actions never fall through.
const shortcuts = (() => {
  const entries = [];
  const byChord = new Map();
  const paint = () => currentTab() === "paint" && typeof maskEditor !== "undefined";
  const viewer = () => state.screen === "workspace" && !!state.run && !maskEditor.corpusActive();
  function add(chords, label, run, enabled = viewer, repeat = false) {
    for (const chord of chords) {
      if (byChord.has(chord)) throw new Error(`Duplicate shortcut: ${chord}`);
      const entry = { chord, label, run, enabled, repeat }; entries.push(entry); byChord.set(chord, entry);
    }
  }
  add(["space"], "Play / pause", togglePlay, () => viewer() && !(paint() && maskEditor.isDirty()));
  for (const [prefix, amount] of [["", 1], ["shift+", 10], ["ctrl+", 100]]) {
    add([prefix + "arrowleft"], `Previous ${amount} frame${amount === 1 ? "" : "s"}`, () => step(-amount), viewer, true);
    add([prefix + "arrowright"], `Next ${amount} frame${amount === 1 ? "" : "s"}`, () => step(amount), viewer, true);
  }
  add(["["], "Previous flagged frame", () => jump("flag", -1));
  add(["]"], "Next flagged frame", () => jump("flag", 1));
  add([","], "Previous low-IoU frame", () => jump("iou", -1));
  add(["."], "Next low-IoU frame", () => jump("iou", 1));
  for (let n = 1; n <= 9; n++) add([String(n)], `Toggle layer ${n}`, () => toggleLayer(n - 1));
  add(["f"], "Raw / flat image", () => $("#toggle-raw").click(), () => viewer() || paint());
  add(["o"], "Mask opacity", () => maskEditor.cycleOpacity(), paint);
  add(["0"], "Fit view", fitView, () => viewer() || paint());
  add(["m"], "Review note", () => { showTab("inspect"); $("#note-comment").focus(); });
  add(["k"], "Fitter starts", computeStarts);
  for (const [key, brush, label] of [["b",255,"Worm"],["e",0,"Background"],["i",127,"Ignore"]]) add([key], `${label} brush`, () => maskEditor.setBrush(brush), paint);
  add(["-"], "Smaller brush", () => maskEditor.resizeBrush(-1), paint, true);
  add(["="], "Larger brush", () => maskEditor.resizeBrush(1), paint, true);
  for (const [key, name] of [["w","network"],["c","classical"],["t","raw_threshold"],["v","saved"]]) add([key], `Preview ${name.replaceAll("_", " ")} proposal`, () => maskEditor.selectProposal(name), paint);
  add(["a"], "Apply proposal", () => maskEditor.applyProposal(), paint);
  for (const [key, name] of [["h","fill_holes"],["l","largest_component"],["d","dilate"],["r","erode"],["u","mask_fit"]]) add([key], `Refine: ${name.replaceAll("_", " ")}`, () => maskEditor.refine(name), paint);
  add(["n"], "Next labeling target", () => maskEditor.next(), paint);
  add(["p"], "Previously visited labeling target", () => maskEditor.previous(), paint);
  add(["s"], "Save mask / label", () => maskEditor.save(), paint);
  add(["enter"], "Save + next", () => maskEditor.saveNext(), paint);
  add(["z","ctrl+z","meta+z"], "Undo draft", () => maskEditor.undoStroke(), paint);
  function chord(event) {
    let key = event.key === " " ? "space" : event.key.toLowerCase();
    // Shift for letter case is not a different accelerator; real modifiers are.
    const shift = event.shiftKey && (!/^[a-z]$/.test(key) || event.ctrlKey || event.metaKey || event.altKey);
    return (event.ctrlKey ? "ctrl+" : "") + (event.metaKey ? "meta+" : "") + (event.altKey ? "alt+" : "") + (shift ? "shift+" : "") + key;
  }
  function dispatch(event) {
    const target = event.target instanceof Element ? event.target : document.activeElement;
    if (event.defaultPrevented || event.isComposing || document.querySelector("dialog[open]")) return;
    if (target && target.closest("input,select,textarea,[contenteditable]:not([contenteditable=false])")) return;
    if (target && target.closest("button,a") && [" ","Enter"].includes(event.key)) return;
    const action = byChord.get(chord(event));
    if (!action || !action.enabled()) return;
    event.preventDefault();
    if (event.repeat && !action.repeat) return;
    Promise.resolve().then(action.run).catch(error => setStatus(error.message, "error"));
  }
  function labelControls() {
    const controls = {"#toggle-raw":"f", "#fit-view":"0", "#starts":"k", "#play":"space", "#prev":"arrowleft", "#next":"arrowright", "#mask-opacity":"o", "#mask-apply-proposal":"a", "#mask-prev":"p", "#mask-next":"n", "#mask-save":"s", "#mask-save-next":"enter", "#mask-stroke-undo":"z"};
    for (const [brush,key] of [[255,"b"],[0,"e"],[127,"i"]]) controls[`[data-mask-brush="${brush}"]`] = key;
    for (const [source,key] of [["network","w"],["classical","c"],["raw_threshold","t"],["saved","v"]]) controls[`[data-mask-proposal="${source}"]`] = key;
    for (const [method,key] of [["fill_holes","h"],["largest_component","l"],["dilate","d"],["erode","r"],["mask_fit","u"]]) controls[`[data-mask-refine="${method}"]`] = key;
    for (const [selector,key] of Object.entries(controls)) {
      const action = byChord.get(key);
      for (const node of document.querySelectorAll(selector)) {
        node.setAttribute("aria-keyshortcuts", key === "space" ? "Space" : key === "enter" ? "Enter" : key.startsWith("arrow") ? "Arrow" + key.slice(5,6).toUpperCase() + key.slice(6) : key.toUpperCase());
        node.dataset.shortcut = action.chord;
        if (!node.title) node.title = `${action.label} (${action.chord})`;
        if (node.tagName === "BUTTON" && / · (?:[A-Z0-9]|Enter)$/.test(node.textContent)) node.textContent = node.textContent.replace(/ · (?:[A-Z0-9]|Enter)$/, ` · ${key === "enter" ? "Enter" : key.toUpperCase()}`);
      }
    }
  }
  function renderHelp() {
    const node = $("#drawer-shortcuts"); node.replaceChildren();
    for (const action of entries) {
      const row = document.createElement("div"); row.className = "shortcut-line";
      const key = document.createElement("kbd"); key.textContent = action.chord;
      const label = document.createElement("span"); label.textContent = action.label;
      row.append(key, label); node.append(row);
    }
  }
  return { init() { window.addEventListener("keydown", dispatch); labelControls(); renderHelp(); }, dispatch, chord, renderHelp, entries };
})();
