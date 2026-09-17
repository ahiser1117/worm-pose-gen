"use strict";
window.workflowExport = (() => {
  let busy=false, generation=0, lastSource=null;
  const host=()=>document.querySelector('#workflow-export-surface');
  const draft=()=>typeof maskEditor!=='undefined'&&maskEditor.isDirty();
  const activeJobs=()=>(state.jobs||[]).filter(j=>j.spec?.workspace===state.runName&&['queued','running'].includes(j.state));
  function controls(){
    const h=host(), b=h?.querySelector('[data-export]');if(!b)return;
    const corpus=typeof maskEditor!=='undefined'&&maskEditor.corpusActive?.();
    const pending=(typeof editPending==='function'&&editPending())||(typeof maskEditor!=='undefined'&&maskEditor.canMutate&&!maskEditor.canMutate());
    b.disabled=busy||!isWorkspace()||!serverIsApp()||draft()||corpus||pending||activeJobs().length>0;
    h.querySelector('[data-export-draft]').textContent=draft()?'Unsaved mask draft: save or discard it in Paint before exporting.':corpus?'Close the corpus label in Paint before exporting this workspace.':pending?'Wait for the current edit or save to finish.':'Unsaved mask drafts: none.';
    h.querySelector('[data-export-jobs]').textContent=`${activeJobs().length} running or queued workspace jobs`;
  }
  async function refresh(){
    const h=host();if(!h)return;const token=++generation;controls();
    const source=currentSourceKey();if(source!==lastSource){lastSource=source;h.querySelector('[data-export-result]').textContent='';h.querySelector('[data-export-summary]').textContent='Checking review status…';}
    h.querySelector('[data-export-context]').textContent=isWorkspace()?`Workspace: ${state.runName} · Whole workspace · frames ${frameOfRow(0)}–${frameOfRow(rowCount()-1)} (${rowCount()} sampled frames).`: 'Open a workspace to export poses.';
    const entry=typeof workspaceEntry==='function'?workspaceEntry(state.runName):null;
    h.querySelector('[data-export-destination]').textContent=`Destination: ${entry?.path||state.runName||'workspace'}/exports/<name>.parquet`;
    if(!isWorkspace()){h.querySelector('[data-export-summary]').textContent='Review counts are available after opening a workspace.';return;}
    try{const summary=await window.workflowInspect?.summary();if(token!==generation)return;h.querySelector('[data-export-summary]').textContent=summary?`${summary.stale_frames??'Unknown'} stale · ${summary.unprocessed_frames} unprocessed · ${summary.unreviewed_flagged_frames} unreviewed flagged frames.`:'Review counts unavailable. You can still export the saved workspace.';}
    catch(e){if(token===generation)h.querySelector('[data-export-summary]').textContent=`Review counts unavailable: ${e.message}. You can still export the saved workspace.`;}
  }
  async function run(){
    controls();if(host().querySelector('[data-export]').disabled)return;
    if(typeof maskEditor!=='undefined'&&!maskEditor.beforeMutation())return;
    const field=host().querySelector('[data-export-name]'), name=field.value.trim(), error=host().querySelector('[data-export-error]');
    if(!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/.test(name)){error.textContent='Enter a name starting with a letter or digit; use letters, digits, dots, underscores or hyphens (up to 80 characters).';field.setAttribute('aria-invalid','true');field.focus();return;}
    error.textContent='';field.removeAttribute('aria-invalid');
    const source=currentSourceKey(), workspace=state.runName;busy=true;controls();const out=host().querySelector('[data-export-result]');out.textContent='Capturing a named snapshot and writing Parquet…';
    try{const result=await post(`/api/workspaces/${encodeURIComponent(workspace)}/export`,{name});if(source!==currentSourceKey())return;out.innerHTML=`<p>Exported ${result.rows} rows from snapshot <b>${escapeHtml(result.snapshot)}</b>.</p><p>${escapeHtml(result.path)}</p><a href="${escapeHtml(result.download_url)}" download>Download Parquet · ${escapeHtml(result.name)}.parquet</a>`;setStatus('Named snapshot and Parquet export complete.','ok');}
    catch(e){if(source===currentSourceKey())out.textContent=`Export failed: ${e.message}. Existing names cannot be overwritten; choose a new name if it already exists, then try again.`;}
    finally{busy=false;controls();}
  }
  function init(){
    const tab=document.querySelector('#tab-export'), main=document.querySelector('#workspace-main');if(!tab||!main)return;
    tab.innerHTML='<p class="note">Review the workspace and create a named export in the main panel.</p>';
    const h=document.createElement('section');h.id='workflow-export-surface';h.className='export-workflow';h.hidden=true;
    h.setAttribute('aria-labelledby','export-heading');
    h.innerHTML='<h2 id="export-heading">Export poses</h2><p data-export-context></p><h3>Review before exporting</h3><ul class="export-checklist"><li><div><p data-export-summary>Checking review status…</p><p>Unresolved flags do not prevent export.</p></div><button type="button" data-inspect>Inspect attention segments</button></li><li><p data-export-draft></p><button type="button" data-paint>Open Paint</button></li><li><p data-export-jobs></p><button type="button" data-jobs>Open Jobs</button></li></ul><p>Export includes saved current poses and provenance as Parquet, with a matching named snapshot of configuration, masks and review records. Accept alternatives in Compare first to include them.</p><label for="export-name">Export / snapshot name</label><input id="export-name" type="text" data-export-name placeholder="reviewed-poses-v1" maxlength="80" aria-describedby="export-name-help export-name-error"><p id="export-name-help">Each name creates a new export and snapshot. Existing names cannot be overwritten.</p><p id="export-name-error" data-export-error role="alert"></p><p data-export-destination></p><div class="row wrap"><button type="button" class="primary" data-export>Create export</button><button type="button" data-refresh>Refresh status</button></div><div data-export-result role="status"></div>';
    main.append(h);h.querySelector('[data-export]').onclick=run;h.querySelector('[data-refresh]').onclick=refresh;
    h.querySelector('[data-inspect]').onclick=()=>{showTab('inspect');window.workflowInspect?.next();};
    h.querySelector('[data-paint]').onclick=()=>showTab('paint');h.querySelector('[data-jobs]').onclick=()=>openRightTab('jobs');
    h.querySelector('[data-export-name]').oninput=()=>{h.querySelector('[data-export-error]').textContent='';h.querySelector('[data-export-name]').removeAttribute('aria-invalid');};
    window.addEventListener('workflow:task',event=>{const active=event.detail?.task==='export'&&(!event.detail.screen||event.detail.screen==='workspace');h.hidden=!active;main.classList.toggle('export-task-active',active);});
    for(const event of ['workflow:edit-state','workflow:mask-state'])window.addEventListener(event,controls);
    for(const event of ['workflow:source','workflow:changed','workflow:jobs','workflow:task','workflow:review'])window.addEventListener(event,refresh);refresh();
  }
  return {init,refresh};
})();
