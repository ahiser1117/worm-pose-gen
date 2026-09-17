"use strict";
window.workflowCompare = (() => {
  let mode = "all", sideBySide = false, active = false;
  function refresh() {
    const host = document.querySelector("#workflow-compare"); if (!host) return;
    const detail = shownDetail(0) || shownDetail(1);
    host.querySelector(".compare-range").textContent = detail ? `Comparing exact frames ${detail.frames[0]}–${detail.frames[1]} · playhead ${frameOfRow(state.row)}` : "Show a candidate set below to compare it with current poses.";
    host.querySelectorAll("[data-overlay]").forEach(b => { b.setAttribute("aria-pressed", String(mode === b.dataset.overlay)); b.disabled = b.dataset.overlay === "A" ? !shownDetail(0) : b.dataset.overlay === "B" ? !shownDetail(1) : false; });
    host.querySelector("[data-range]").disabled = !detail;
    renderSide();
  }
  function visible(slot) { return !active || mode === "all" || mode === slot; }
  function renderSide() {
    const host = document.querySelector("#compare-side"); if (!host) return;
    host.hidden = !sideBySide || !active;
    document.querySelector("#workspace-main")?.classList.toggle("compare-side-active", sideBySide && active);
    if (!sideBySide || !active) return;
    const img = state.showRaw && state.imageRaw ? state.imageRaw : state.image;
    ["Current", "A", "B"].forEach((label, i) => {
      const cell = host.children[i], c = cell.querySelector("canvas"), g = c.getContext("2d");
      cell.hidden = i > 0 && !shownDetail(i - 1);
      if (cell.hidden) return;
      const d = state.decoded;
      cell.querySelector("span").textContent = `${label} · frame ${frameOfRow(state.row)}`;
      g.clearRect(0, 0, c.width, c.height);
      if (!d?.width) return;
      const scale = Math.min(c.width / d.width, c.height / d.height);
      g.save(); g.translate((c.width-d.width*scale)/2,(c.height-d.height*scale)/2); g.scale(scale,scale);
      if(img)g.drawImage(img,0,0);
      let points = state.frame?.pose?.centerline_xy;
      if(i) { const e=shownDetail(i-1)?.byRow?.get(state.row); points=e?.chosen != null ? e.candidates[e.chosen]?.centerline_xy : null; if(e?.mirrored&&points)points=points.slice().reverse(); }
      if(points?.length) {g.strokeStyle=i?slotCss(i-1):"#ff50a5";g.lineWidth=2/scale;g.beginPath();points.forEach((p,j)=>j?g.lineTo(...p):g.moveTo(...p));g.stroke();g.fillStyle="#ffd23c";g.fillRect(points[0][0]-3/scale,points[0][1]-3/scale,6/scale,6/scale);}
      g.restore(); cell.querySelector("span").textContent=`${label} · frame ${frameOfRow(state.row)}${points?'':' · no pose'}`;
    });
  }
  function init() {
    const tab=document.querySelector("#tab-review"); if(!tab)return;
    const host=document.createElement("section");host.id="workflow-compare";
    host.innerHTML='<p class="compare-range"></p><div class="row wrap"><span>Overlay</span><button data-overlay="all">Current + A + B</button><button data-overlay="Current">Current</button><button data-overlay="A">A</button><button data-overlay="B">B</button><button data-range>Go to range start</button></div><div class="row wrap"><label><input type="checkbox" id="compare-side-toggle"> Side by side</label><button data-try>Try another method</button><button data-next>Next segment needing review</button></div><div id="compare-side" hidden>'+['Current','A','B'].map(x=>`<figure><span>${x}</span><canvas width="640" height="480" aria-label="${x} pose at current playhead"></canvas></figure>`).join('')+'</div>';
    tab.prepend(host);
    const panes=host.querySelector("#compare-side"), stage=document.querySelector("#stage");
    if(stage)stage.before(panes);
    window.addEventListener("workflow:task",e=>{active=e.detail?.task==="review";draw();refresh();});
    host.querySelectorAll('[data-overlay]').forEach(b=>b.onclick=()=>{mode=b.dataset.overlay;draw();refresh();});
    host.querySelector('#compare-side-toggle').onchange=e=>{sideBySide=e.target.checked;refresh();};
    host.querySelector('[data-range]').onclick=()=>{const d=shownDetail(0)||shownDetail(1);if(d)showRow(d.rows[0],{keepView:true});};
    host.querySelector('[data-try]').onclick=()=>{const d=shownDetail(0)||shownDetail(1);if(d){setRegionRows(d.rows[0],d.rows[1],'comparison range');setRerunScope('selection');}showTab('rerun');};
    host.querySelector('[data-next]').onclick=()=>{showTab('inspect');window.workflowInspect?.next();};
    for(const event of ['workflow:source','workflow:selection','workflow:changed','workflow:task','workflow:candidates'])window.addEventListener(event,refresh);
    refresh();
  }
  return {init,refresh,renderSide,visible};
})();
