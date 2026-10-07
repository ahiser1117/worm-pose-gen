// The Labeling page: see docs/APP_SIMPLIFICATION.md. Placeholder until wave 2 builds it.

import {el} from "./api.js";

export function mount(section, ctx) {
  section.append(el("p", {class: "empty"}, "Labeling is being rebuilt."));
}
