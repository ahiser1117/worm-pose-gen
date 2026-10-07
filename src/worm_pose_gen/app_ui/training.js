// The Training page (docs/APP_SIMPLIFICATION.md, section 4): three tabs for
// the current setup, chosen in the header.
//
//   #training/models               the model picker table with row actions and the running jobs
//   #training/datasets             datasets, their splits and readiness; Freeze benchmark
//   #training/train[/<model ref>]  the Train form, starting from that model
//
// Each tab is a module (training_models.js, training_datasets.js,
// training_train.js) exporting render(container, page, params), which may
// return a function that stops its polling when the tab is left. `page` holds
// what the tabs share: ctx, the setups, the current setup, compute choices and
// go(tab, params) for moving between tabs.

import {api, el} from "./api.js";
import * as modelsTab from "./training_models.js";
import * as datasetsTab from "./training_datasets.js";
import * as trainTab from "./training_train.js";

const TABS = {models: ["Models", modelsTab], datasets: ["Datasets", datasetsTab], train: ["Train", trainTab]};
const SETUP_KEY = "worm-pose.training.setup";

const page = {ctx: null, setups: [], setup: null, compute: null, go: null};
let root, body, tabs, stop = null, currentTab = "models", currentParams = "";

function remember(ref) {
  try { localStorage.setItem(SETUP_KEY, ref); } catch { /* private window: not remembered */ }
}

function remembered() {
  try { return localStorage.getItem(SETUP_KEY); } catch { return null; }
}

export async function mount(section, ctx) {
  page.ctx = ctx;
  page.go = (tab, params = "") => ctx.navigate("training", params ? `${tab}/${params}` : tab);
  tabs = el("nav", {class: "tr-tabs", role: "tablist", "aria-label": "Training"},
    Object.entries(TABS).map(([key, [label]]) => el("button", {role: "tab", dataset: {tab: key}, onclick: () => page.go(key)}, label)));
  body = el("div", {class: "tr-body"});
  root = el("div", {class: "tr-page"}, tabs, body);
  section.append(root);
  const [{setups}, compute] = await Promise.all([api("/api/library/setups"), api("/api/compute").catch(() => null)]);
  page.setups = setups;
  page.compute = compute;
  const saved = remembered();
  page.setup = setups.find((s) => s.ref === saved) || setups.find((s) => s.ref.startsWith("lab:")) || setups[0] || null;
}

function header() {
  if (!page.setups.length) return [];
  const select = el("select", {class: "tr-setup", "aria-label": "Setup", onchange: async () => {
    page.setup = page.setups.find((s) => s.ref === select.value);
    remember(page.setup.ref);
    render();
  }}, page.setups.map((s) => el("option", {value: s.ref, selected: s.ref === page.setup?.ref}, `${s.name} (${s.ref})`)));
  return [el("label", {class: "inline tr-setup-label"}, "Setup", select)];
}

async function render() {
  stop?.();
  stop = null;
  for (const button of tabs.querySelectorAll("button")) button.setAttribute("aria-selected", String(button.dataset.tab === currentTab));
  if (!page.setup) {
    body.replaceChildren(el("p", {class: "empty"}, "No setup in the libraries yet. A setup (a microscope) is created when recordings are added."));
    return;
  }
  // Refresh the setup itself (its defaults change from the Models tab).
  try {
    const fresh = await api(`/api/library/setups/${page.setup.ref}`);
    page.setup = fresh;
    page.setups = page.setups.map((s) => (s.ref === fresh.ref ? fresh : s));
  } catch { /* keep what we have */ }
  page.ctx.setHeader("training", {context: header()});
  const container = el("div", {class: `tr-tab tr-tab-${currentTab}`});
  body.replaceChildren(container);
  try {
    const result = await TABS[currentTab][1].render(container, page, currentParams);
    stop = typeof result === "function" ? result : null;
  } catch (error) {
    container.replaceChildren(el("p", {class: "note error"}, `Could not load ${TABS[currentTab][0]}: ${error.message}`));
  }
}

export async function show(params = "") {
  const [tab, ...rest] = (params || "").split("/");
  currentTab = TABS[tab] ? tab : "models";
  currentParams = rest.join("/");
  await render();
}

export function hide() {
  stop?.();
  stop = null;
}
