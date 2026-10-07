// The model picker: one table of the setup's models with their inputs,
// outputs and benchmark numbers (docs/APP_SIMPLIFICATION.md, section 4).
// Owned by the Training page; the Workspace opens it for "Change model".
//
//   const ref = await openModelPicker(ctx, {setup: "lab:nir-flv", role: "mask", current: "lab:nir-hand284"});
//   // ref: the chosen model ref, or null when cancelled
//
// Placeholder until wave 2 (Training) builds it: a plain list from the library.

import {api, el, query} from "./api.js";

export async function openModelPicker(ctx, {setup, role, current = null}) {
  const {models = []} = await api(`/api/library/models${query({setup})}`);
  const usable = models.filter((card) => (card.outputs || []).includes(role === "body" ? "ap" : "mask"));
  return new Promise((resolve) => {
    const dialog = el("dialog", {class: "model-picker"});
    const list = el("div", {class: "list"}, usable.map((card) => el("div", {
      class: "item", "aria-current": String(card.ref === current),
      onclick: () => { dialog.close(); resolve(card.ref); },
    }, card.name || card.ref)));
    dialog.append(el("h2", {}, "Choose a model"), list, el("div", {class: "row"}, el("button", {onclick: () => { dialog.close(); resolve(null); }}, "Cancel")));
    dialog.addEventListener("close", () => { dialog.remove(); resolve(null); });
    document.body.append(dialog);
    dialog.showModal();
  });
}
