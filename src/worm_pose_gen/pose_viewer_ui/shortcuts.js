"use strict";

// A chord has one meaning in every task. Disabled actions never fall through.
const shortcuts = (() => {
  const entries = [];
  const byChord = new Map();
  // Mask tools act on the workspace Masks task or on Paint, whichever is showing.
  const masks = () => state.screen === "workspace" && currentTab() === "masks";
  const labeling = () => paintScreen.isOpen();
  const editing = () => masks() || labeling();
  const editor = () => labeling() ? paintScreen : maskEditor;
  const viewer = () => state.screen === "workspace" && !!state.run;
  function add(chords, label, run, enabled = viewer, repeat = false) {
    for (const chord of chords) {
      if (byChord.has(chord)) throw new Error(`Duplicate shortcut: ${chord}`);
      const entry = { chord, label, run, enabled, repeat }; entries.push(entry); byChord.set(chord, entry);
    }
  }
  add(["space"], "Play / pause", togglePlay, () => viewer() && !(masks() && maskEditor.isDirty()));
  for (const [prefix, amount] of [["", 1], ["shift+", 10], ["ctrl+", 100]]) {
    add([prefix + "arrowleft"], `Previous ${amount} frame${amount === 1 ? "" : "s"}`, () => step(-amount), viewer, true);
    add([prefix + "arrowright"], `Next ${amount} frame${amount === 1 ? "" : "s"}`, () => step(amount), viewer, true);
  }
  add(["["], "Previous flagged frame", () => jump("flag", -1));
  add(["]"], "Next flagged frame", () => jump("flag", 1));
  add([","], "Previous low-IoU frame", () => jump("iou", -1));
  add(["."], "Next low-IoU frame", () => jump("iou", 1));
  for (let n = 1; n <= 9; n++) add([String(n)], `Toggle layer ${n}`, () => toggleLayer(n - 1));
  add(["f"], "Raw / flat image", () => labeling() ? paintScreen.toggleRaw() : $("#toggle-raw").click(), () => viewer() || labeling());
  add(["o"], "Mask opacity", () => editor().cycleOpacity(), editing);
  add(["0"], "Fit view", () => labeling() ? paintScreen.fitView() : fitView(), () => viewer() || labeling());
  add(["m"], "Review note", () => { showTab("inspect"); $("#note-comment").focus(); });
  add(["k"], "Fitter starts", computeStarts);
  for (const [key, brush, label] of [["b",255,"Worm"],["e",0,"Background"],["i",127,"Ignore"]]) add([key], `${label} brush`, () => editor().setBrush(brush), editing);
  add(["-"], "Smaller brush", () => editor().resizeBrush(-1), editing, true);
  add(["="], "Larger brush", () => editor().resizeBrush(1), editing, true);
  for (const [key, name] of [["w","network"],["c","classical"],["t","raw_threshold"],["v","saved"]]) add([key], `Preview ${name.replaceAll("_", " ")} proposal`, () => editor().selectProposal(name), editing);
  add(["a"], "Apply proposal", () => editor().applyProposal(), editing);
  for (const [key, name] of [["h","fill_holes"],["l","largest_component"],["d","dilate"],["r","erode"],["u","mask_fit"]]) add([key], `Refine: ${name.replaceAll("_", " ")}`, () => editor().refine(name), editing);
  add(["n"], "Next entry in the label group", () => paintScreen.next(), labeling);
  add(["p"], "Previous entry in the label group", () => paintScreen.previous(), labeling);
  add(["s"], "Save mask / label", () => editor().save(), editing);
  add(["enter"], "Save label + next unlabeled", () => paintScreen.saveNext(), labeling);
  add(["z","ctrl+z","meta+z"], "Undo draft", () => editor().undoStroke(), editing);
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
    const controls = {"#toggle-raw":"f", "#fit-view":"0", "#starts":"k", "#play":"space", "#prev":"arrowleft", "#next":"arrowright", "#mask-opacity":"o", "#mask-apply-proposal":"a", "#mask-save":"s", "#mask-stroke-undo":"z",
      "#paint-raw":"f", "#paint-fit":"0", "#paint-opacity":"o", "#paint-apply":"a", "#paint-prev":"p", "#paint-next":"n", "#paint-save":"s", "#paint-save-next":"enter", "#paint-undo":"z"};
    for (const prefix of ["mask", "paint"]) {
      for (const [brush,key] of [[255,"b"],[0,"e"],[127,"i"]]) controls[`[data-${prefix}-brush="${brush}"]`] = key;
      for (const [source,key] of [["network","w"],["classical","c"],["raw_threshold","t"],["saved","v"]]) controls[`[data-${prefix}-proposal="${source}"]`] = key;
      for (const [method,key] of [["fill_holes","h"],["largest_component","l"],["dilate","d"],["erode","r"],["mask_fit","u"]]) controls[`[data-${prefix}-refine="${method}"]`] = key;
    }
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
