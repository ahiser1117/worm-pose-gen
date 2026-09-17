"use strict";

// Traversal never invents a source pool or wraps at exhaustion. Visits are
// independent of the temporal playhead and retain canonical recording identity.
const paintNavigation = (() => {
  let explicitPool=null,manifestId=null,cursor=null,visits=[],visitIndex=-1,busy=false,generation=0;
  let editorState={busy:false,network:undefined},catalogNetwork,configured=false,startupPoolPending=false,startupManifestId=null,startupAttempted=false;
  const manifests=new Map();
  const node=id=>document.getElementById(id);
  const value=(id,fallback)=>node(id)?.value||fallback;
  const currentTarget=()=>typeof maskEditor!=='undefined'?maskEditor.getTarget():null;
  const mode=()=>value('mask-next-mode','sequential');
  const kind=()=>value('mask-next-pool',isWorkspace()?'workspace':'recordings');
  function status(message) {if(node('mask-next-status'))node('mask-next-status').textContent=message;}
  function pool() {
    if(explicitPool)return structuredClone(explicitPool);
    const selected=kind();
    if(selected==='workspace'||selected==='selection') {
      if(!isWorkspace())throw new Error('Open a workspace or choose a recording pool.');
      const result={workspace:state.runName};
      if(selected==='selection') {
        const frames=state.run?.series?.frame_index;
        if(!state.region||!frames||state.region.first<0||state.region.last<state.region.first||state.region.last>=frames.length)throw new Error('Select a valid frame range first.');
        result.frames=frames.slice(state.region.first,state.region.last+1);
      }
      if(!(result.frames||state.run?.series?.frame_index)?.length)throw new Error('This workspace has no frames to traverse.');
      return result;
    }
    if(selected==='manifest'){if(!manifestId)throw new Error('Load a manifest first.');return {};}
    const paths=value('mask-recordings','').split(/\r?\n/).map(v=>v.trim()).filter(Boolean);
    if(paths.length)return {recordings:paths.map(recording=>({recording,dataset:'/img_nir'}))};
    const selectedRecordings=typeof corpusUI!=='undefined'&&corpusUI.getPool?corpusUI.getPool():null;
    if(selectedRecordings?.recordings?.length)return selectedRecordings;
    if(mode()==='browse')return {};
    throw new Error('Choose at least one recording for this labeling pool.');
  }
  function unavailable() {
    if(!serverIsApp())return 'Labeling traversal requires the app server.';
    if(busy||editorState.busy)return 'Wait for the current operation to finish.';
    const target=currentTarget(),selectedMode=mode();
    if(selectedMode==='uncertain'&&(editorState.network===false||(!target&&catalogNetwork===false)))return 'Network-uncertain requires an available checkpoint.';
    if(selectedMode==='queue'&&!manifestId)return 'Load a manifest to use Queue mode.';
    if(kind()==='manifest'&&selectedMode!=='queue')return 'Manifest pool is used by Queue mode; choose a workspace or recordings for this mode.';
    let selectedPool;
    try{selectedPool=pool();}catch(error){return error.message;}
    if(target?.workspace&&selectedPool.workspace!==target.workspace)return 'Workspace correction stays in its workspace or selected range.';
    if(target&&!target.workspace&&selectedPool.workspace)return 'Return to the workspace before traversing workspace frames.';
    if(selectedMode==='sequential'&&(!Number.isInteger(Number(value('mask-stride','1')))||Number(value('mask-stride','1'))<1))return 'Enter a positive integer stride.';
    if(selectedMode==='uncertain'&&(!Number.isInteger(Number(value('mask-candidates','6')))||Number(value('mask-candidates','6'))<1||Number(value('mask-candidates','6'))>256))return 'Enter a candidate count from 1 to 256.';
    if(selectedMode==='queue') {
      const manifest=manifests.get(manifestId);
      if(manifest?.progress?.remaining===0)return 'This manifest queue is complete.';
      if(Array.isArray(manifest?.entries)) {
        let entries=manifest.entries;
        if(selectedPool.workspace&&target?.recording){const frames=new Set(selectedPool.frames||state.run?.series?.frame_index||[]);entries=entries.filter(entry=>entry.target?.recording===target.recording&&entry.target?.dataset===target.dataset&&frames.has(entry.target?.frame));}
        if(!entries.length)return 'No manifest entries belong to this labeling pool.';
        if(entries.every(entry=>entry.error))return `Queue sources are unavailable: ${entries[0].error}`;
      }
    }
    return '';
  }
  function disabled(id,reason) {const field=node(id);if(field){field.disabled=!!reason;field.title=reason||'';field.setAttribute('aria-disabled',String(!!reason));}}
  function refreshControls(update) {
    if(update)editorState={...editorState,...update};
    const target=currentTarget();
    if(startupPoolPending&&target?.workspace){explicitPool=null;if(node('mask-next-pool'))node('mask-next-pool').value='workspace';startupPoolPending=false;generation++;}
    const waiting=busy||editorState.busy?'Wait for the current operation to finish.':!serverIsApp()?'Labeling traversal requires the app server.':'';
    const selectedMode=mode(),selectedKind=kind(),networkUnavailable=editorState.network===false||(!target&&catalogNetwork===false);
    disabled('mask-next-mode',waiting);disabled('mask-next-pool',waiting);
    disabled('mask-stride',waiting||(selectedMode!=='sequential'?'Stride applies only to Sequential mode.':''));
    disabled('mask-candidates',waiting||(selectedMode!=='uncertain'?'Candidate count applies only to Network-uncertain mode.':networkUnavailable?'Choose an available network checkpoint first.':''));
    disabled('mask-recordings',waiting||(selectedKind!=='recordings'?'Recording paths apply only to the Recordings pool.':''));
    disabled('mask-manifest',waiting||(selectedMode!=='queue'?'Manifest loading applies only to Queue mode.':''));
    disabled('mask-manifest-load',waiting||(selectedMode!=='queue'?'Choose Queue mode to load a manifest.':!value('mask-manifest','').trim()?'Enter a manifest path first.':''));
    const reason=unavailable();disabled('mask-next',reason);disabled('mask-prev',waiting||(visitIndex<1?'No previous visited labeling frame.':''));
    const uncertain=node('mask-next-mode')?.querySelector('option[value="uncertain"]');if(uncertain)uncertain.disabled=networkUnavailable;
    let help=node('mask-next-help');if(!help&&node('mask-next-status')){help=document.createElement('div');help.id='mask-next-help';help.className='note';help.setAttribute('aria-live','polite');node('mask-next-status').insertAdjacentElement('afterend',help);}
    if(help)help.textContent=reason||({sequential:'Stride advances through sampled frames in the declared pool.',uncertain:'Candidates limits the distinct unlabeled frames scored by the network.',random:'Choose a random unlabeled frame within the declared pool.',queue:'Queue follows the loaded manifest and preserves its split pledges.',browse:'Browse follows the current saved-label filters without wrapping.'}[selectedMode]||'');
    return !reason;
  }
  function record(target) {
    if(!target)return;
    const copy=structuredClone(target),identity=t=>JSON.stringify([t.workspace||null,t.recording||t.source_path,t.dataset||t.dataset_path,t.frame??t.frame_index,t.sample_id||null]);
    if(visitIndex>=0&&identity(visits[visitIndex])===identity(copy)){refreshControls();return;}
    visits=visits.slice(0,visitIndex+1);visits.push(copy);visitIndex=visits.length-1;refreshControls();
  }
  async function next() {
    const reason=unavailable();if(reason){status(reason);refreshControls();return false;}
    if(!await maskEditor.requestLeave({preserveTarget:true}))return false;
    busy=true;refreshControls();
    try {
      const selectedPool=pool(),target=maskEditor.getTarget(),token=generation,selectedMode=mode();
      const stride=selectedMode==='sequential'?Number(value('mask-stride','1')):1,candidates=selectedMode==='uncertain'?Number(value('mask-candidates','6')):6;
      const result=await post('/api/labeling/next',{mode:selectedMode,pool:selectedPool,current:target,stride,candidates,manifest_id:manifestId,cursor,filters:typeof corpusUI!=='undefined'&&corpusUI.getFilters?corpusUI.getFilters():{}});
      if(token!==generation||JSON.stringify(maskEditor.getTarget())!==JSON.stringify(target)){status('Navigation context changed; the old next-frame result was discarded.');return false;}
      if(result.progress&&manifests.has(manifestId))manifests.get(manifestId).progress=result.progress;
      if(result.blocked){status(result.reason||'The next source recording is unavailable.');return false;}
      if(result.exhausted){status(result.reason||'This labeling pool is complete.');return false;}
      if(!result.target)throw new Error('The server returned no next target.');
      if(target?.workspace&&result.target.workspace!==target.workspace)throw new Error('The next result is outside the active correction workspace.');
      if(!await maskEditor.openTarget(result.target,{navigation:true}))return false;
      cursor=result.cursor||null;status(result.progress?progressLabel(result.progress):'Next labeling frame loaded.');return true;
    }catch(error){status(error.message);return false;}finally{busy=false;refreshControls();}
  }
  async function previous() {
    if(busy||editorState.busy||visitIndex<1){status('No previous visited labeling frame is available.');return false;}
    busy=true;refreshControls();
    try{const index=visitIndex-1;if(await maskEditor.openTarget(visits[index],{recordVisit:false})){visitIndex=index;status('Previous visited labeling frame.');return true;}return false;}
    finally{busy=false;refreshControls();}
  }
  function progressLabel(progress) {return `${progress.remaining??'?'} remaining · ${progress.labeled??'?'} labeled · ${progress.total??'?'} total`;}
  function selectManifest(manifest,startup=false) {
    const id=manifest.manifest_id||manifest.id;if(!id)throw new Error('Manifest response has no identity.');
    manifests.set(id,manifest);manifestId=id;cursor=null;generation++;
    if(node('mask-manifest')&&manifest.path)node('mask-manifest').value=manifest.path;
    if(node('mask-next-mode'))node('mask-next-mode').value='queue';
    const target=currentTarget();
    if(!(target?target.workspace:isWorkspace()&&state.run)){explicitPool=null;if(node('mask-next-pool'))node('mask-next-pool').value='manifest';startupPoolPending=startup;}
    else if(!['workspace','selection'].includes(kind())){explicitPool=null;if(node('mask-next-pool'))node('mask-next-pool').value='workspace';}
    status(`${manifest.name||'Manifest queue'}${manifest.progress?' · '+progressLabel(manifest.progress):''}${manifest.errors?.length?' · '+manifest.errors.length+' unavailable source(s); locate and register them.':''}`);refreshControls();
  }
  async function loadManifest() {
    if(busy||editorState.busy||mode()!=='queue'||!value('mask-manifest','').trim())return false;
    busy=true;configured=true;startupPoolPending=false;refreshControls();
    try{selectManifest(await post('/api/labeling/manifests',{path:value('mask-manifest','').trim()}));return true;}
    catch(error){status(error.message);return false;}finally{busy=false;refreshControls();}
  }
  function onCatalog(info) {
    if(Object.prototype.hasOwnProperty.call(info||{},'fallback_checkpoint'))catalogNetwork=!!info.fallback_checkpoint;
    const rows=Array.isArray(info?.labeling_manifests)?info.labeling_manifests:[];
    for(const manifest of rows)if(manifest.id||manifest.manifest_id)manifests.set(manifest.id||manifest.manifest_id,manifest);
    if(!startupManifestId&&rows.length)startupManifestId=rows[0].id||rows[0].manifest_id;
    if(!manifestId&&!configured&&rows.length)selectManifest(rows[0],true);
    refreshControls();
  }
  // Boot-only launcher bridge: a manifest declares corpus work independently of
  // whichever workspace the viewer happened to open most recently.
  async function startStartupQueue() {
    if(startupAttempted||!startupManifestId)return false;
    startupAttempted=true;
    const showPaint=()=>{if(typeof showTask==='function')showTask('paint');else if(typeof showTab==='function')showTab('paint');};
    if(maskEditor.isDirty()){showPaint();status('Startup queue is ready. Save or discard the existing draft before opening it.');return false;}
    if(isWorkspace()&&!currentTarget()&&maskEditor.onFrame)await maskEditor.onFrame(true);
    if(maskEditor.isDirty()){showPaint();status('Startup queue did not replace the current draft.');return false;}
    const initialTarget=JSON.stringify(currentTarget()),manifest=manifests.get(startupManifestId),id=startupManifestId;
    startupPoolPending=false;busy=true;refreshControls();
    try {
      const result=await post('/api/labeling/next',{mode:'queue',pool:{},manifest_id:id,current:null,stride:1,candidates:6});
      if(maskEditor.isDirty()||JSON.stringify(currentTarget())!==initialTarget){showPaint();status('The viewing context changed; startup queue navigation was cancelled.');return false;}
      if(result.progress&&manifest)manifest.progress=result.progress;
      if(!result.target){
        manifestId=id;explicitPool=null;cursor=null;generation++;
        if(node('mask-next-mode'))node('mask-next-mode').value='queue';if(node('mask-next-pool'))node('mask-next-pool').value='manifest';
        showPaint();status(result.reason||(result.exhausted?'The startup manifest queue is complete.':'The startup queue source is unavailable.'));return false;
      }
      // openTarget captures the workspace context before its corpus setup changes
      // any traversal settings, so Return to workspace restores the original pool.
      if(!await maskEditor.openTarget({...result.target,manifest_id:result.target.manifest_id||id}))return false;
      manifestId=id;explicitPool=null;cursor=result.cursor||null;generation++;
      if(node('mask-next-mode'))node('mask-next-mode').value='queue';if(node('mask-next-pool'))node('mask-next-pool').value='manifest';
      if(node('mask-manifest')&&manifest?.path)node('mask-manifest').value=manifest.path;
      showPaint();status(`${manifest?.name||'Startup queue'}${result.progress?' · '+progressLabel(result.progress):''}`);return true;
    }catch(error){showPaint();status(`Startup queue could not open: ${error.message}`);return false;}
    finally{busy=false;refreshControls();}
  }
  function manifestForTarget(target){if(mode()!=='queue'||!target)return null;const entries=manifests.get(manifestId)?.entries;return entries?.some(entry=>entry.target?.recording===target.recording&&entry.target?.dataset===target.dataset&&entry.target?.frame===target.frame)?manifestId:null;}
  const contextFields=['mask-next-mode','mask-next-pool','mask-recordings','mask-manifest','mask-stride','mask-candidates'];
  function captureContext(){return {explicitPool:structuredClone(explicitPool),manifestId,cursor:structuredClone(cursor),visits:structuredClone(visits),visitIndex,fields:Object.fromEntries(contextFields.map(id=>[id,node(id)?.value]))};}
  function restoreContext(context){if(!context)return;explicitPool=context.explicitPool;manifestId=context.manifestId;cursor=context.cursor;visits=context.visits;visitIndex=context.visitIndex;generation++;for(const [id,value] of Object.entries(context.fields))if(node(id)&&value!==undefined)node(id).value=value;refreshControls();}
  function beginCorpus(options={}){cursor=null;visits=[];visitIndex=-1;startupPoolPending=false;generation++;if(!options.preserveNavigation){explicitPool=null;if(node('mask-next-pool'))node('mask-next-pool').value='recordings';if(node('mask-next-mode'))node('mask-next-mode').value='sequential';}refreshControls();}
  function setPool(selectedPool){explicitPool=structuredClone(selectedPool);cursor=null;configured=true;startupPoolPending=false;generation++;if(selectedPool&&node('mask-next-pool'))node('mask-next-pool').value=selectedPool.workspace?(selectedPool.frames?'selection':'workspace'):'recordings';if(selectedPool?.recordings&&node('mask-recordings'))node('mask-recordings').value=selectedPool.recordings.map(record=>record.recording).join('\n');refreshControls();}
  function init() {
    if(node('mask-next'))node('mask-next').onclick=next;
    if(node('mask-prev'))node('mask-prev').onclick=previous;
    if(node('mask-manifest-load'))node('mask-manifest-load').onclick=loadManifest;
    for(const id of ['mask-next-mode','mask-next-pool','mask-recordings','mask-stride','mask-candidates','mask-manifest'])if(node(id))node(id).addEventListener(['mask-next-mode','mask-next-pool'].includes(id)?'change':'input',()=>{cursor=null;configured=true;startupPoolPending=false;if(['mask-next-pool','mask-recordings'].includes(id))explicitPool=null;generation++;refreshControls();});
    if(state.info)onCatalog(state.info);refreshControls();
  }
  return {init,next,previous,record,pool,captureContext,restoreContext,beginCorpus,onCatalog,startStartupQueue,refreshControls,manifestForTarget,manifest:()=>manifestId,setPool,resetCursor(){cursor=null;generation++;refreshControls();}};
})();
