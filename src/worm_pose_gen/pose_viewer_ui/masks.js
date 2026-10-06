"use strict";

// Workspace mask correction (the Masks task): the current frame's override,
// which fitting reads, with its proposals, refinement and optional copy into
// the training corpus. Corpus labeling itself is Paint (paint.js).
// PNG labels use 0 background / 127 ignore / 255 worm at this UI boundary.
const maskEditor = (() => {
  let data=null,pixels=null,saved=null,history=[],dirty=false,busy=false;
  let target=null,key=null,generation=0,draftGeneration=0,frameTimer=null,frameResolve=null;
  let brush=255,diameter=12,active=true,painting=false,last=null,cursor=null,opacity=.45;
  let proposal=null,proposalName=null,proposalRequest=0,refineRequest=0,refining=false;
  let frozenSave=null,leavePromise=null,probability=null;
  const overlay=document.createElement('canvas'),preview=document.createElement('canvas');
  let predicted=null;
  const node=id=>document.getElementById(id);
  const value=(id,fallback)=>node(id)?.value||fallback;
  const checked=(id,fallback=false)=>node(id)?node(id).checked:fallback;
  const text=(id,message)=>{if(node(id))node(id).textContent=message;};
  const listen=(id,event,fn)=>{if(node(id))node(id).addEventListener(event,fn);};
  const enabled=()=>active&&checked('mask-enable');
  const pendingEdit=()=>typeof editPending==='function'&&editPending();
  const supported=()=>serverIsApp()&&isWorkspace();
  const currentKey=()=>`${currentSourceKey()}:${currentFrameIndex()}`;
  const status=message=>text('mask-status',message);
  const ready=()=>supported()&&!!pixels&&!busy&&!pendingEdit()&&key===currentKey();
  const displayLayer=()=>typeof LAYERS==='undefined'?null:LAYERS.find(item=>item.id==='editable_mask');
  const displaySettings=()=>displayLayer()||{on:checked('mask-visible',true),alpha:opacity};
  const displayAvailable=()=>active&&!!pixels&&key===currentKey()&&!state.playing&&!state.timeline?.dragging;
  function syncDisplayControls() {
    const settings=displaySettings();
    if(node('mask-visible'))node('mask-visible').checked=settings.on;
    const select=node('mask-opacity'),percent=String(Math.round(settings.alpha*100));
    if(select){
      if(!Array.from(select.options).some(option=>option.value===percent)){
        select.querySelector('[data-custom-opacity]')?.remove();
        const option=document.createElement('option');option.value=percent;option.textContent=percent+'%';option.dataset.customOpacity='true';select.append(option);
      }
      select.value=percent;
    }
    const entry=document.querySelector('.layer[data-layer="editable_mask"]');
    if(entry){entry.querySelector('input[type=checkbox]').checked=settings.on;entry.querySelector('input[type=range]').value=settings.alpha;}
    if(typeof renderLegend==='function')renderLegend();
    if(typeof renderLayerAvailability==='function')renderLayerAvailability();
  }
  function setDisplay(settings) {
    const entry=displayLayer();
    if(settings.on!==undefined){if(entry)entry.on=!!settings.on;if(node('mask-visible'))node('mask-visible').checked=!!settings.on;}
    if(settings.alpha!==undefined){opacity=Math.max(0,Math.min(1,Number(settings.alpha)));if(entry)entry.alpha=opacity;}
    syncDisplayControls();draw();
  }
  function controls() {
    const ok=ready();
    for(const id of ['mask-save','mask-save-refit','mask-clear','mask-to-corpus','mask-refit-frame','mask-refit-stretch','mask-stroke-undo','mask-discard','mask-clear-draft','mask-apply-proposal'])if(node(id))node(id).disabled=!ok;
    if(node('mask-enable'))node('mask-enable').disabled=!supported()||busy;
    if(node('mask-clear'))node('mask-clear').disabled=!ok||!data?.has_override||dirty;
    if(node('mask-stroke-undo'))node('mask-stroke-undo').disabled=!ok||!history.length;
    if(node('mask-discard'))node('mask-discard').disabled=!ok||(!dirty&&!frozenSave);
    if(node('mask-apply-proposal'))node('mask-apply-proposal').disabled=!ok||!proposal;
    if(node('mask-save-refit'))node('mask-save-refit').disabled=!ok||!state.region;
    if(node('mask-refit-frame'))node('mask-refit-frame').disabled=!ok||dirty||!!frozenSave;
    if(node('mask-refit-stretch'))node('mask-refit-stretch').disabled=!ok||dirty||!!frozenSave||!state.region;
    if(node('mask-also-corpus'))node('mask-also-corpus').disabled=busy||!!frozenSave;
    if(node('mask-corpus-split'))node('mask-corpus-split').disabled=busy||!!frozenSave||!!data?.pledged_split||!checked('mask-also-corpus');
    for(const button of document.querySelectorAll('[data-mask-brush]')){button.disabled=!ok;button.classList.toggle('active',Number(button.dataset.maskBrush)===brush);}
    for(const button of document.querySelectorAll('[data-mask-proposal]')){const source=sourceName(button.dataset.maskProposal);button.disabled=!ok||data?.capabilities?.[source]===false;button.classList.toggle('active',button.dataset.maskProposal===proposalName);}
    for(const button of document.querySelectorAll('[data-mask-refine]'))button.disabled=!ok||refining;
    if(node('mask-size'))node('mask-size').disabled=!ok;
    if(node('mask-threshold'))node('mask-threshold').disabled=!ok||data?.capabilities?.network===false;
    for(const option of node('mask-saved-source')?.options||[])option.disabled=data?.capabilities?.[option.value==='corpus'?'saved_corpus':'saved_workspace']===false;
    text('mask-refit-target',state.region?`Refit scope: frames ${state.run?.series.frame_index[state.region.first]}–${state.run?.series.frame_index[state.region.last]}. Save changes only the mask on frame ${target?.frame??'…'}.`:'Select a range in Inspect to save and refit it.');
    text('mask-target',target?`Workspace mask · frame ${target.frame} · ${state.runName}${data?.pledged_split?' · corpus copy pledged '+data.pledged_split:''}`:'Open a workspace to correct its masks.');
    syncDisplayControls();
    window.dispatchEvent(new CustomEvent('workflow:mask-state'));
  }
  function refreshOverlay() {if(!data||!pixels)return;maskTools.render(overlay,data.width,data.height,pixels);if(proposal)maskTools.render(preview,data.width,data.height,proposal,true);controls();draw();}
  function draftStatus() {if(data)status(`Frame ${data.frame}: ${dirty?'unsaved draft':data.has_override?'saved override':'predicted mask (no override)'}${data.stale?' · pose needs refit':''}`);}
  async function install(payload,token,wanted) {
    const [decoded,image,base]=await Promise.all([decodeGray(payload.mask),loadImage(payload.image),decodeGray(payload.base_mask)]);
    if(token!==generation)return false;
    if(!decoded||!image)throw new Error('Mask or source image is missing.');
    const width=payload.width||image.width,height=payload.height||image.height;
    if(decoded.length!==width*height)throw new Error('Mask dimensions do not match the source image.');
    target=structuredClone(payload.target||wanted);data={...payload,width,height,frame:payload.frame??target.frame};
    pixels=maskTools.normalize(decoded);saved=pixels.slice();history=[];dirty=false;frozenSave=null;proposal=null;proposalName=null;draftGeneration++;
    predicted=null;if(base){predicted=document.createElement('canvas');predicted.width=width;predicted.height=height;const c=predicted.getContext('2d'),img=c.createImageData(width,height);for(let i=0;i<base.length;i++)if(base[i]>200){img.data[i*4+1]=220;img.data[i*4+2]=255;img.data[i*4+3]=110;}c.putImageData(img,0,0);}
    if(node('mask-corpus-split')&&payload.pledged_split)node('mask-corpus-split').value=payload.pledged_split;
    refreshOverlay();draftStatus();return true;
  }
  function invalidate() {
    clearTimeout(frameTimer);if(frameResolve){frameResolve(false);frameResolve=null;}
    generation++;proposalRequest++;refineRequest++;refining=false;key=null;data=null;pixels=null;saved=null;history=[];dirty=false;painting=false;proposal=null;proposalName=null;frozenSave=null;target=null;controls();
  }
  async function loadFrame(force=false) {
    if(!supported()){invalidate();status('Read-only source: import as a workspace to correct masks.');return false;}
    const wanted=currentKey();if(!force&&key===wanted)return true;if(dirty||busy||frozenSave)return false;
    const wantedTarget={workspace:state.runName,frame:currentFrameIndex()},token=++generation;key=wanted;pixels=null;data=null;controls();
    try{return await install(await post('/api/labeling/frame',{target:wantedTarget}),token,wantedTarget);}
    catch(error){if(token===generation){key=null;status(error.message);controls();}return false;}
  }
  function onFrame(force=false) {
    clearTimeout(frameTimer);if(frameResolve){frameResolve(false);frameResolve=null;}
    if(force)return loadFrame(true);controls();if(state.playing||state.timeline?.dragging)return Promise.resolve(false);
    return new Promise(resolve=>{frameResolve=resolve;frameTimer=setTimeout(()=>{frameResolve=null;if(state.playing||state.timeline?.dragging){resolve(false);return;}loadFrame().then(resolve);},180);});
  }
  function beforeMutation() {
    if(dirty||busy||pendingEdit()||frozenSave){setStatus(busy||pendingEdit()?'Wait for the current save to finish.':'Save or discard the mask draft first.','error');return false;}return true;
  }
  function leave() {if(dirty||busy||frozenSave){status('Save or discard the mask draft before navigating.');return false;}invalidate();return true;}
  async function requestLeave(options={}) {
    if(leavePromise)return leavePromise;
    leavePromise=(async()=>{
      if(busy||pendingEdit()){status('Wait for the current save to finish.');return false;}
      if(dirty||frozenSave) {
        const choice=typeof requestDraftDecision==='function'?await requestDraftDecision():'stay';
        if(choice==='save'){if(!(await save()).ok)return false;}
        else if(choice==='discard')discardDraft();else return false;
      }
      if(!options.preserveTarget)invalidate();return true;
    })();
    try{return await leavePromise;}finally{leavePromise=null;}
  }
  function pushUndo() {history.push(pixels.slice());if(history.length>30)history.shift();}
  function mutated() {draftGeneration++;dirty=pixels.some((v,i)=>v!==saved[i]);frozenSave=null;refreshOverlay();draftStatus();}
  function undoStroke() {if(!ready()||!history.length)return false;pixels=history.pop();mutated();return true;}
  function discardDraft() {if(!pixels||busy)return false;pixels=saved.slice();history=[];dirty=false;frozenSave=null;draftGeneration++;refreshOverlay();draftStatus();return true;}
  function clearDraft() {if(!ready())return false;pushUndo();pixels.fill(0);mutated();return true;}
  // A save is frozen with its destinations, so a retry repeats only the failed one.
  async function save(options={}) {
    if(!ready())return {ok:false,destinations:{},error:'The mask is not ready to save.'};
    busy=true;painting=false;last=null;draftGeneration++;controls();
    if(!frozenSave) {
      const destinations=options.corpusOnly?['corpus']:checked('mask-also-corpus')?['workspace','corpus']:['workspace'];
      frozenSave={target:structuredClone(target),pixels:pixels.slice(),mask:maskTools.encode(data.width,data.height,pixels),workspaceRevision:data.revision,corpusRevision:data.corpus_revision??0,split:value('mask-corpus-split',''),destinations,results:{}};
    }
    const transaction=frozenSave;
    for(const destination of transaction.destinations) {
      if(transaction.results[destination]?.ok)continue;
      try {
        if(destination==='workspace') {
          const result=await post(editsUrl('mask'),{frame:transaction.target.frame,mask:transaction.mask,revision:transaction.workspaceRevision});
          transaction.results.workspace={ok:true,result};data.has_override=true;data.stale=true;if(data.capabilities)data.capabilities.saved_workspace=true;
          const detail=result.mask&&typeof result.mask==='object'?result.mask:await api(editsUrl('mask',`frame=${transaction.target.frame}`)).catch(()=>null);
          if(detail){data.revision=detail.revision;transaction.workspaceRevision=detail.revision;}
          saved=transaction.pixels.slice();dirty=false;
          if(typeof applyEditResponse==='function')await applyEditResponse(result,{preserveMaskDraft:true}).catch(error=>status(`Mask saved; viewer refresh failed: ${error.message}`));
        } else {
          const result=await post('/api/labeling/save',{target:transaction.target,mask:transaction.mask,revision:transaction.corpusRevision,...(transaction.split?{split:transaction.split}:{})});
          transaction.results.corpus={ok:true,result};
          if(result.sample){data.sample=result.sample;data.corpus_revision=result.sample.revision;transaction.corpusRevision=result.sample.revision;if(data.capabilities)data.capabilities.saved_corpus=true;}
          if(typeof corpusUI!=='undefined')await corpusUI.refresh().catch(()=>{});
        }
      }catch(error){transaction.results[destination]={ok:false,error:error.message};}
    }
    const ok=transaction.destinations.every(destination=>transaction.results[destination]?.ok);
    text('mask-save-status',transaction.destinations.map(destination=>`${destination==='workspace'?'Workspace mask':'Corpus label'}: ${transaction.results[destination]?.ok?'saved':transaction.results[destination]?.error||'not saved'}`).join(' · '));
    if(ok){frozenSave=null;history=[];status(transaction.destinations.length===1&&transaction.destinations[0]==='corpus'?'Corpus label saved; the workspace draft is separate.':'Mask saved. Refit creates pose candidates for review.');}
    else status('Some selected saves failed. Retry Save to complete only the failed destination.');
    busy=false;controls();draw();return {ok,destinations:{...transaction.results}};
  }
  async function saveRefit() {
    if(!ready()||!state.region)return false;
    const region={...state.region},source=currentSourceKey(),frame=currentFrameIndex();
    const result=await save();
    if(!result.ok||!result.destinations.workspace?.ok||source!==currentSourceKey())return false;
    if(frame!==currentFrameIndex()||state.region?.first!==region.first||state.region?.last!==region.last){status('Mask saved. Selection changed during saving; choose the intended range and prepare its refit.');return false;}
    status('Mask saved; current pose is stale. Run a fit, then compare candidates before accepting.');
    return refit(true);
  }
  async function removeOverride() {
    if(!ready()||dirty||!data.has_override)return {ok:false};busy=true;controls();
    try {const result=await api(editsUrl('mask')+`?frame=${data.frame}&revision=${encodeURIComponent(data.revision||'')}`,{method:'DELETE'});if(typeof applyEditResponse==='function')await applyEditResponse(result);busy=false;invalidate();await onFrame(true);status('Override removed. The automatic mask is restored; undo this saved operation in History.');return {ok:true};}
    catch(error){status(error.message);return {ok:false};}finally{busy=false;controls();}
  }
  function sourceName(name) {return name==='saved'?(value('mask-saved-source','override')==='corpus'?'saved_corpus':'saved_workspace'):name;}
  function updateThreshold() {
    const threshold=Number(value('mask-threshold','.5'));text('mask-threshold-value',threshold.toFixed(2));
    if(probability&&proposalName==='network'){proposal=probability.map(v=>v>=threshold*255?255:0);refreshOverlay();}
  }
  async function selectProposal(name) {
    if(!ready())return false;const source=sourceName(name);if(data.capabilities?.[source]===false){text('mask-proposal-status','This proposal is unavailable for the current frame.');return false;}
    const token=generation,request=++proposalRequest;proposalName=name;proposal=null;probability=null;controls();text('mask-proposal-status',`Loading ${source} preview…`);
    try {
      const response=await post('/api/labeling/proposals',{target,source,request_id:request});
      const [mask,prob]=await Promise.all([decodeGray(response.mask),decodeGray(response.probability)]);
      if(token!==generation||request!==proposalRequest)return false;
      if(!mask&&!prob)throw new Error('The proposal has no mask.');
      if((mask||prob).length!==pixels.length)throw new Error('Proposal dimensions do not match this frame.');
      probability=prob;proposal=maskTools.normalize(mask);if(probability)updateThreshold();else refreshOverlay();
      text('mask-proposal-status',`${source.replaceAll('_',' ')} preview · Apply changes the draft.`);return true;
    }catch(error){if(token===generation&&request===proposalRequest){proposal=null;probability=null;text('mask-proposal-status',error.message);controls();}return false;}
  }
  function applyProposal(event) {
    if(!ready()||!proposal)return false;
    pushUndo();pixels=maskTools.combine(pixels,proposal,maskTools.combineMode(event,value('mask-combine','replace')));mutated();return true;
  }
  async function refine(method) {
    if(!ready()||refining)return false;
    const token=generation,revision=draftGeneration,request=++refineRequest,mask=maskTools.encode(data.width,data.height,pixels);refining=true;controls();text('mask-refine-status',`Running ${method.replaceAll('_',' ')}…`);
    try {
      const response=await post('/api/labeling/refine',{target:structuredClone(target),mask,method,request_id:request,draft_generation:revision});const labels=maskTools.normalize(await decodeGray(response.mask));
      if(token!==generation||request!==refineRequest)return false;
      if(revision!==draftGeneration){text('mask-refine-status','The draft changed while refinement ran; its result was discarded.');return false;}
      if(!labels||labels.length!==pixels.length)throw new Error('Refinement dimensions do not match this frame.');
      pushUndo();pixels=labels;mutated();text('mask-refine-status',`${method.replaceAll('_',' ')} applied${response.info?' · '+JSON.stringify(response.info):''}`);return true;
    }catch(error){if(token===generation)text('mask-refine-status',error.message);return false;}
    finally{if(token===generation&&request===refineRequest){refining=false;controls();}}
  }
  function setBrush(value) {if(!ready())return false;brush=typeof value==='string'?({worm:255,background:0,ignore:127}[value]??Number(value)):value;controls();draw();return true;}
  function resizeBrush(delta) {if(!ready())return false;diameter=Math.max(1,Math.min(100,diameter+delta));if(node('mask-size'))node('mask-size').value=diameter;text('mask-size-value',diameter+' px');draw();return true;}
  function stroke(a,b) {maskTools.stroke(pixels,data.width,data.height,a,b,diameter,brush);mutated();}
  function paint(event) {
    if(!enabled()||!ready()||event.button!==0||event.shiftKey||event.altKey||event.ctrlKey||event.metaKey)return;
    event.preventDefault();event.stopImmediatePropagation();if(state.playing)togglePlay();painting=true;last=toImage(event.clientX,event.clientY);cursor=last;pushUndo();canvas.setPointerCapture(event.pointerId);stroke(last,last);
  }
  function drawLayer() {
    if(!displayAvailable())return;
    if(predicted&&checked('mask-predicted'))ctx.drawImage(predicted,0,0);
    const settings=displaySettings();
    ctx.save();ctx.globalAlpha=settings.alpha;
    if(settings.on)ctx.drawImage(overlay,0,0);
    if(proposal&&checked('mask-proposal-preview',true))ctx.drawImage(preview,0,0);
    ctx.restore();
    if(enabled()&&cursor){ctx.save();ctx.beginPath();ctx.arc(cursor.x,cursor.y,diameter/2,0,Math.PI*2);ctx.strokeStyle=maskTools.brushColor(brush);ctx.lineWidth=1.5/state.view.scale;ctx.stroke();ctx.restore();}
  }
  async function refit(selection) {if(!beforeMutation())return false;if(typeof prepareMaskRefit==='function')return prepareMaskRefit(selection?'selection':'current');status('Rerun controls are not available.');return false;}
  async function ensureSavedForRerun() {
    if(busy||pendingEdit())return false;if(!dirty&&!frozenSave)return true;
    const choice=await new Promise(resolve=>{const dialog=document.createElement('dialog'),message=document.createElement('p');message.textContent='Rerun needs a saved mask. Save this draft and use it?';dialog.append(message);for(const [answer,label] of [['save','Save and use mask'],['masks','Return to Masks']]){const button=document.createElement('button');button.textContent=label;button.onclick=()=>{dialog.close();dialog.remove();resolve(answer);};dialog.append(button);}dialog.addEventListener('cancel',event=>{event.preventDefault();dialog.close();dialog.remove();resolve('masks');},{once:true});document.body.append(dialog);dialog.showModal();});
    if(choice==='save')return (await save()).ok;showTab('masks');return false;
  }
  function cycleOpacity() {const current=displaySettings().alpha;setDisplay({alpha:current>.4?.2:current>0?0:.45});return displaySettings().alpha;}
  function setActive(value) {active=!!value;if(active&&state.playing)togglePlay();painting=false;cursor=null;controls();draw();}
  function init() {
    listen('mask-save-refit','click',saveRefit);
    window.addEventListener('workflow:selection',controls);
    listen('mask-enable','change',()=>{if(enabled()&&state.playing)togglePlay();onFrame();draw();});
    for(const button of document.querySelectorAll('[data-mask-brush]'))button.onclick=()=>setBrush(Number(button.dataset.maskBrush));
    for(const button of document.querySelectorAll('[data-mask-proposal]'))button.onclick=async event=>{if(await selectProposal(button.dataset.maskProposal)){if(event.shiftKey||event.altKey||event.ctrlKey||event.metaKey)applyProposal(event);}};
    for(const button of document.querySelectorAll('[data-mask-refine]'))button.onclick=()=>refine(button.dataset.maskRefine);
    listen('mask-size','input',event=>{diameter=Math.max(1,Math.min(100,Number(event.target.value)));text('mask-size-value',diameter+' px');draw();});
    listen('mask-visible','change',event=>setDisplay({on:event.target.checked}));
    for(const id of ['mask-predicted','mask-proposal-preview'])listen(id,'change',draw);
    listen('mask-opacity','change',event=>setDisplay({alpha:Number(event.target.value)/100}));
    listen('mask-save','click',()=>save());listen('mask-clear','click',removeOverride);listen('mask-clear-draft','click',clearDraft);
    listen('mask-stroke-undo','click',undoStroke);listen('mask-discard','click',discardDraft);listen('mask-to-corpus','click',()=>save({corpusOnly:true}));
    listen('mask-refit-frame','click',()=>refit(false));listen('mask-refit-stretch','click',()=>refit(true));listen('mask-also-corpus','change',controls);
    listen('mask-threshold','input',updateThreshold);listen('mask-saved-source','change',()=>{if(proposalName==='saved')selectProposal('saved');});listen('mask-apply-proposal','click',applyProposal);
    canvas.addEventListener('pointerdown',paint,true);
    canvas.addEventListener('pointermove',event=>{cursor=toImage(event.clientX,event.clientY);if(painting){event.stopImmediatePropagation();stroke(last,cursor);last=cursor;}else if(enabled())draw();},true);
    for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,event=>{if(painting){painting=false;last=null;event.stopImmediatePropagation();controls();draw();}},true);
    canvas.addEventListener('pointerleave',()=>{cursor=null;draw();});canvas.addEventListener('contextmenu',event=>{if(enabled())event.preventDefault();});
    window.addEventListener('beforeunload',event=>{if(dirty||busy||frozenSave){event.preventDefault();event.returnValue='';}});controls();
  }
  return {init,onFrame,invalidate,leave,requestLeave,beforeMutation,canMutate:()=>!dirty&&!busy&&!pendingEdit()&&!frozenSave,ensureSavedForRerun,undoStroke,draw:drawLayer,isDirty:()=>dirty||!!frozenSave,getTarget:()=>target?structuredClone(target):null,save,setBrush,resizeBrush,selectProposal,applyProposal,refine,clearDraft,discardDraft,cycleOpacity,setActive,setDisplay,syncDisplayControls,displayAvailable,refreshControls:controls};
})();
