// The app shell: three pages behind one header, switched by the URL hash
// (#workspace, #labeling, #training, optionally followed by /<params> that
// the page interprets). Each page is a module exporting
//   mount(section, ctx)   once, on first visit
//   show(params)          on every visit (params: the hash after the page)
//   hide()                when another page is shown; may return false to
//                         refuse leaving (e.g. an unsaved draft)
// ctx gives every page the same services: the server config (/api/config:
// dev mode, libraries, compute), navigation, the header slots and toasts.
// Pages talk to each other only through ctx.navigate and the server.

import {api} from "./api.js";

const PAGES = ["workspace", "labeling", "training"];
const modules = {}, mounted = {};
let current = null, toastTimer = null;

const ctx = {
  config: null,
  get dev() { return !!ctx.config?.dev; },
  navigate(page, params = "") {
    const hash = `#${page}${params ? `/${params}` : ""}`;
    if (location.hash === hash) show(page, params);
    else location.hash = hash;
  },
  // The header's middle (context) and right (actions) slots belong to the visible page.
  setHeader(page, {context = [], actions = []} = {}) {
    if (page !== current) return;
    document.getElementById("header-context").replaceChildren(...context);
    document.getElementById("header-actions").replaceChildren(...actions);
  },
  toast(message, kind = "") {
    const node = document.getElementById("toast");
    node.textContent = message;
    node.className = `toast ${kind}`;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, kind === "error" ? 8000 : 4000);
  },
};

async function load(page) {
  if (!modules[page]) modules[page] = await import(`./${page}.js`);
  if (!mounted[page]) {
    mounted[page] = true;
    await modules[page].mount(document.getElementById(`page-${page}`), ctx);
  }
  return modules[page];
}

async function show(page, params) {
  if (!PAGES.includes(page)) page = "workspace";
  if (current && current !== page) {
    const leaving = modules[current];
    if (leaving?.hide && leaving.hide() === false) {
      history.replaceState(null, "", `#${current}`);
      return;
    }
  }
  const previous = current;
  current = page;
  for (const name of PAGES) {
    document.getElementById(`page-${name}`).hidden = name !== page;
    document.querySelector(`#page-tabs [data-page="${name}"]`).setAttribute("aria-selected", String(name === page));
  }
  if (previous !== page) ctx.setHeader(page, {});
  try {
    const module = await load(page);
    await module.show?.(params);
  } catch (error) {
    ctx.toast(`Could not open ${page}: ${error.message}`, "error");
    throw error;
  }
}

function route() {
  const [page, ...rest] = location.hash.replace(/^#/, "").split("/");
  show(page || "workspace", rest.join("/"));
}

async function start() {
  try {
    ctx.config = await api("/api/config");
  } catch (error) {
    ctx.config = {dev: false};
    ctx.toast(`Could not read the app configuration: ${error.message}`, "error");
  }
  document.body.classList.toggle("dev", ctx.dev);
  for (const tab of document.querySelectorAll("#page-tabs [data-page]")) {
    tab.addEventListener("click", () => ctx.navigate(tab.dataset.page));
  }
  window.addEventListener("hashchange", route);
  route();
}

start();
