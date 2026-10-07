// The Workspace page (docs/APP_SIMPLIFICATION.md, section 2): the
// Recordings screen and one workspace per recording. The hash after
// #workspace says which:
//   (nothing)                     the Recordings screen (workspace_home.js)
//   <workspace>                   that workspace (workspace_view.js)
//   <workspace>/stitch/<queue>    back from Labeling with a Relabel queue
//                                 done: stitch the stretch between its labels
// The other modules: workspace_analyse.js (models and Run on),
// workspace_timeline.js, workspace_mask.js (Edit mask), workspace_draw.js
// and workspace_dev.js (--dev only).

import {RecordingsScreen} from "./workspace_home.js";
import {WorkspaceView} from "./workspace_view.js";

let ctx = null, home = null, view = null;

export function mount(section, context) {
  ctx = context;
  section.classList.add("ws-page");
  home = new RecordingsScreen(section, ctx, {onOpen: (name) => ctx.navigate("workspace", encodeURIComponent(name))});
  view = new WorkspaceView(section, ctx, {onHome: () => ctx.navigate("workspace")});
}

export async function show(params = "") {
  const [name, action, id] = params.split("/").map((part) => decodeURIComponent(part));
  if (view.name && name !== view.name && !view.node.hidden && !view.canLeave()) {
    history.replaceState(null, "", `#workspace/${encodeURIComponent(view.name)}`);
    return;
  }
  if (!name) {
    view.hide();
    await home.show();
    return;
  }
  home.hide();
  await view.open(name);
  if (action === "stitch" && id) {
    // Once: a reload of the page must not stitch the same queue again.
    history.replaceState(null, "", `#workspace/${encodeURIComponent(name)}`);
    await view.stitch(id);
  }
}

export function hide() {
  if (!view.canLeave()) return false;
  view.hide();
  home.hide();
  return true;
}
