"use strict";

// A target's editable labels, proposal preview and saved copies are independent.
// PNG labels use 0 background / 127 ignore / 255 worm at this UI boundary.
const maskEditor = (() => {
  let data=null,pixels=null,saved=null,history=[],dirty=false,busy=false;
  let target=null,key=null,generation=0,draftGeneration=0,frameTimer=null,frameResolve=null;
  let brush=255,diameter=12,active=true,painting=false,last=null,cursor=null,opacity=.45;
  let proposal=null,proposalName=null,proposalRequest=0,refineRequest=0,refining=false;
  let frozenSave=null,workspaceContext=null,leavePromise=null,suppressVisit=false;
  const overlay=document.createElement('canvas'),preview=document.createElement('canvas');
  let predicted=null;
  const node=id=>document.getElementById(id);
  const value=(id,fallback)=>node(id)?.value||fallback;
  const checked=(id,fallback=false)=>node(id)?node(id).checked:fallback;
  const text=(id,message)=>{if(node(id))node(id).textContent=message;};
  const listen=(id,event,fn)=>{if(node(id))node(id).addEventListener(event,fn);};
  const corpusActive=()=>!!target&&!target.workspace;
  const enabled=()=>active&&checked('mask-enable');
  const pendingEdit=()=>typeof editPending==='function'&&editPending();
  const supported=()=>serverIsApp()&&(isWorkspace()||corpusActive());
  const currentKey=()=>`${currentSourceKey()}:${currentFrameIndex()}`;
  const nav=()=>typeof paintNavigation==='undefined'?null:paintNavigation;
  const status=message=>text('mask-status',message);
  const ready=()=>supported()&&!!pixels&&!busy&&!pendingEdit()&&(corpusActive()||key===currentKey());
  const displayLayer=()=>typeof LAYERS==='undefined'?null:LAYERS.find(item=>item.id==='editable_mask');
  const displaySettings=()=>displayLayer()||{on:checked('mask-visible',true),alpha:opacity};
  const displayAvailable=()=>active&&!!pixels&&(corpusActive()||key===currentKey())&&!state.playing&&!state.timeline?.dragging;
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
    for(const id of ['mask-save','mask-save-refit','mask-save-next','mask-clear','mask-to-corpus','mask-refit-frame','mask-refit-stretch','mask-stroke-undo','mask-discard','mask-clear-draft','mask-apply-proposal','mask-delete-corpus'])if(node(id))node(id).disabled=!ok;
    if(node('mask-enable'))node('mask-enable').disabled=!supported()||busy;
    if(node('mask-save'))node('mask-save').textContent=corpusActive()?'Save corpus label · S':'Save mask · S';
    if(node('mask-clear'))node('mask-clear').disabled=!ok||corpusActive()||!data?.has_override||dirty;
    if(node('mask-to-corpus'))node('mask-to-corpus').disabled=!ok||corpusActive();
    if(node('mask-stroke-undo'))node('mask-stroke-undo').disabled=!ok||!history.length;
    if(node('mask-discard'))node('mask-discard').disabled=!ok||(!dirty&&!frozenSave);
    if(node('mask-apply-proposal'))node('mask-apply-proposal').disabled=!ok||!proposal;
    if(node('mask-save-refit'))node('mask-save-refit').disabled=!ok||corpusActive()||!state.region;
    if(node('mask-refit-frame'))node('mask-refit-frame').disabled=!ok||corpusActive()||dirty||!!frozenSave;
    if(node('mask-refit-stretch'))node('mask-refit-stretch').disabled=!ok||corpusActive()||dirty||!!frozenSave||!state.region;
    if(node('mask-delete-corpus'))node('mask-delete-corpus').disabled=!ok||!data?.sample;
    if(node('mask-also-corpus'))node('mask-also-corpus').disabled=corpusActive()||busy||!!frozenSave;
    if(node('mask-corpus-split'))node('mask-corpus-split').disabled=busy||!!frozenSave||!!data?.pledged_split||(!corpusActive()&&!checked('mask-also-corpus'));
    for(const button of document.querySelectorAll('[data-mask-brush]')){button.disabled=!ok;button.classList.toggle('active',Number(button.dataset.maskBrush)===brush);}
    for(const button of document.querySelectorAll('[data-mask-proposal]')){const source=sourceName(button.dataset.maskProposal);button.disabled=!ok||data?.capabilities?.[source]===false;button.classList.toggle('active',button.dataset.maskProposal===proposalName);}
    for(const button of document.querySelectorAll('[data-mask-refine]'))button.disabled=!ok||refining;
    if(node('mask-size'))node('mask-size').disabled=!ok;
    if(node('mask-threshold'))node('mask-threshold').disabled=!ok||data?.capabilities?.network===false;
    for(const option of node('mask-saved-source')?.options||[])option.disabled=data?.capabilities?.[option.value==='corpus'?'saved_corpus':'saved_workspace']===false;
    if(node('corpus-close'))node('corpus-close').hidden=!corpusActive();
    if(node('mask-corpus-close')){node('mask-corpus-close').hidden=false;node('mask-corpus-close').disabled=!corpusActive()||busy;node('mask-corpus-close').title=corpusActive()?'Restore the workspace viewing context.':'Available while editing a corpus label.';}
    const recording=target&&(target.recording||target.source_path||target.workspace),basename=recording?.split(/[\\/]/).filter(Boolean).pop();
    text('mask-refit-target',state.region?`Refit scope: frames ${state.run?.series.frame_index[state.region.first]}–${state.run?.series.frame_index[state.region.last]}. Save changes only the mask on frame ${target?.frame??data?.frame??'…'}.`:'Select a range in Inspect to save and refit it.');
    text('mask-target',target?`${corpusActive()?'Corpus label':'Workspace mask'} · frame ${target.frame??data?.frame} · ${basename}${target.dataset?' · '+target.dataset:''}${data?.pledged_split?' · pledged '+data.pledged_split:''}`:'Open a workspace or corpus recording to paint.');
    if(node('mask-target'))node('mask-target').title=target?`${recording}${target.dataset?' · '+target.dataset:''} · frame ${target.frame??data?.frame}`:'';
    nav()?.refreshControls({busy:busy||pendingEdit()||(!pixels&&key!==null),network:data?.capabilities?.network});
    syncDisplayControls();
    window.dispatchEvent(new CustomEvent('workflow:mask-state'));
  }
  function renderMask(canvas,labels,isProposal=false) {
    canvas.width=data.width;canvas.height=data.height;
    const c=canvas.getContext('2d'),image=c.createImageData(data.width,data.height);
    for(let i=0;i<labels.length;i++) {
      const v=labels[i],j=i*4;
      image.data[j]=isProposal?40:255;image.data[j+1]=isProposal?220:v===127?205:65;image.data[j+2]=isProposal?255:v===127?40:155;
      image.data[j+3]=v===0?0:255;
    }
    c.putImageData(image,0,0);
  }
  function refreshOverlay() {if(!data||!pixels)return;renderMask(overlay,pixels);if(proposal)renderMask(preview,proposal,true);controls();draw();}
  function draftStatus() {if(data)status(`${corpusActive()?'Corpus label':'Frame '+data.frame}: ${dirty?'unsaved draft':corpusActive()?(data.sample?'saved label':'new corpus draft (not saved)'):data.has_override?'saved override':'predicted mask (no override)'}${data.stale?' · pose needs refit':''}`);}
  function normalize(labels) {return labels?.map(v=>v===128?127:v);}
  async function install(payload,token,wanted) {
    const [decoded,image,raw,base]=await Promise.all([decodeGray(payload.mask||payload.label),loadImage(payload.image),loadImage(payload.image_raw),decodeGray(payload.base_mask)]);
    if(token!==generation)return false;
    if(!decoded||!image)throw new Error('Mask or source image is missing.');
    const width=payload.width||image.width,height=payload.height||image.height;
    if(decoded.length!==width*height)throw new Error('Mask dimensions do not match the source image.');
    target=structuredClone(payload.target||wanted);data={...payload,width,height,frame:payload.frame??target.frame??target.frame_index,_image:image,_raw:raw};
    pixels=normalize(decoded);saved=pixels.slice();history=[];dirty=false;frozenSave=null;proposal=null;proposalName=null;draftGeneration++;
    predicted=null;if(base){predicted=document.createElement('canvas');predicted.width=width;predicted.height=height;const c=predicted.getContext('2d'),img=c.createImageData(width,height);for(let i=0;i<base.length;i++)if(base[i]>200){img.data[i*4+1]=220;img.data[i*4+2]=255;img.data[i*4+3]=110;}c.putImageData(img,0,0);}
    if(node('mask-corpus-split')&&payload.pledged_split)node('mask-corpus-split').value=payload.pledged_split;
    refreshOverlay();draftStatus();return true;
  }
  function invalidate() {
    clearTimeout(frameTimer);if(frameResolve){frameResolve(false);frameResolve=null;}
    generation++;proposalRequest++;refineRequest++;refining=false;key=null;data=null;pixels=null;saved=null;history=[];dirty=false;painting=false;proposal=null;proposalName=null;frozenSave=null;target=null;controls();
  }
  async function loadFrame(force=false) {
    if(corpusActive())return false;
    if(!supported()){invalidate();status('Read-only source: import as a workspace to correct masks.');return false;}
    const wanted=currentKey();if(!force&&key===wanted)return true;if(dirty||busy||frozenSave)return false;
    const wantedTarget={workspace:state.runName,frame:currentFrameIndex()},token=++generation;key=wanted;pixels=null;data=null;controls();
    try {
      // The legacy mask endpoint remains a fallback for older app servers.
      let payload;
      try{if(typeof post!=='function')throw new Error('404');payload=await post('/api/labeling/frame',{target:wantedTarget});}
      catch(error){if(!/404|not found|unknown endpoint/i.test(error.message))throw error;payload=await api(editsUrl('mask',`frame=${wantedTarget.frame}`));}
      const installed=await install(payload,token,wantedTarget);if(installed&&!suppressVisit)nav()?.record(target);return installed;
    }catch(error){if(token===generation){key=null;status(error.message);controls();}return false;}
  }
  function onFrame(force=false) {
    clearTimeout(frameTimer);if(frameResolve){frameResolve(false);frameResolve=null;}
    if(force)return loadFrame(true);controls();if(state.playing||state.timeline?.dragging)return Promise.resolve(false);
    return new Promise(resolve=>{frameResolve=resolve;frameTimer=setTimeout(()=>{frameResolve=null;if(state.playing||state.timeline?.dragging){resolve(false);return;}loadFrame().then(resolve);},180);});
  }
  function beforeMutation() {
    if(dirty||busy||pendingEdit()||corpusActive()||frozenSave){setStatus(corpusActive()?'Return to the workspace before changing its poses.':busy||pendingEdit()?'Wait for the current save to finish.':'Save or discard the mask draft first.','error');return false;}return true;
  }
  // Compatibility for callers not yet migrated to the asynchronous guard.
  function leave() {if(dirty||busy||frozenSave){status('Save or discard the mask draft before navigating.');return false;}invalidate();return true;}
  async function requestLeave(options={}) {
    if(leavePromise)return leavePromise;
    leavePromise=(async()=>{
      if(busy||pendingEdit()){status('Wait for the current save to finish.');return false;}
      if(dirty||frozenSave) {
        const choice=typeof requestDraftDecision==='function'?await requestDraftDecision():await fallbackDecision();
        if(choice==='save'){if(!(await save()).ok)return false;}
        else if(choice==='discard')discardDraft();else return false;
      }
      if(!options.preserveTarget)invalidate();if(options.destination?.kind==='source')workspaceContext=null;return true;
    })();
    try{return await leavePromise;}finally{leavePromise=null;}
  }
  function fallbackDecision() {
    return new Promise(resolve=>{
      const dialog=document.createElement('dialog'),message=document.createElement('p');message.textContent='This frame has unsaved changes. Save before continuing?';dialog.append(message);
      for(const [choice,label] of [['save','Save and continue'],['discard','Discard'],['stay','Stay']]){const button=document.createElement('button');button.textContent=label;button.onclick=()=>{dialog.close();dialog.remove();resolve(choice);};dialog.append(button);}
      dialog.addEventListener('cancel',event=>{event.preventDefault();dialog.close();dialog.remove();resolve('stay');},{once:true});document.body.append(dialog);dialog.showModal();
    });
  }
  function pushUndo() {history.push(pixels.slice());if(history.length>30)history.shift();}
  function mutated() {draftGeneration++;dirty=pixels.some((v,i)=>v!==saved[i]);frozenSave=null;refreshOverlay();draftStatus();}
  function undoStroke() {if(!ready()||!history.length)return false;pixels=history.pop();mutated();return true;}
  function discardDraft() {if(!pixels||busy)return false;pixels=saved.slice();history=[];dirty=false;frozenSave=null;draftGeneration++;refreshOverlay();draftStatus();return true;}
  function clearDraft() {if(!ready())return false;pushUndo();pixels.fill(0);mutated();return true;}
  function encode(labels=pixels) {
    const out=document.createElement('canvas');out.width=data.width;out.height=data.height;const c=out.getContext('2d'),image=c.createImageData(out.width,out.height);
    for(let i=0;i<labels.length;i++){const j=i*4;image.data[j]=image.data[j+1]=image.data[j+2]=labels[i];image.data[j+3]=255;}c.putImageData(image,0,0);return out.toDataURL('image/png');
  }
  async function save(options={}) {
    if(options===true)return removeOverride();
    if(!ready())return {ok:false,destinations:{},error:'The mask is not ready to save.'};
    busy=true;painting=false;last=null;draftGeneration++;controls();
    if(!frozenSave) {
      const destinations=options.corpusOnly?['corpus']:corpusActive()?['corpus']:checked('mask-also-corpus')?['workspace','corpus']:['workspace'];
      frozenSave={target:structuredClone(target),pixels:pixels.slice(),mask:encode(),draftGeneration,workspaceRevision:data.revision,corpusRevision:data.corpus_revision??data.sample?.revision??0,split:value('mask-corpus-split',''),manifest:target.manifest_id||nav()?.manifestForTarget(target),destinations,results:{}};
    }
    const transaction=frozenSave;
    for(const destination of transaction.destinations) {
      if(transaction.results[destination]?.ok)continue;
      try {
        let result;
        if(destination==='workspace') {
          result=await post(editsUrl('mask'),{frame:transaction.target.frame??data.frame,mask:transaction.mask,revision:transaction.workspaceRevision});
          transaction.results.workspace={ok:true,result};data.has_override=true;data.stale=true;if(data.capabilities)data.capabilities.saved_workspace=true;
          // Refresh the optimistic token without replacing the frozen pixels.
          const detail=result.mask&&typeof result.mask==='object'?result.mask:await api(editsUrl('mask',`frame=${transaction.target.frame??data.frame}`)).catch(()=>null);
          if(detail){data.revision=detail.revision;transaction.workspaceRevision=detail.revision;}
          saved=transaction.pixels.slice();dirty=false;
          if(typeof applyEditResponse==='function')await applyEditResponse(result,{preserveMaskDraft:true}).catch(error=>status(`Mask saved; viewer refresh failed: ${error.message}`));
        } else {
          result=await post('/api/labeling/save',{target:transaction.target,mask:transaction.mask,revision:transaction.corpusRevision,...(transaction.split?{split:transaction.split}:{}),...(transaction.manifest?{manifest_id:transaction.manifest}:{})});
          transaction.results.corpus={ok:true,result};const sample=result.sample||result.record;
          if(sample){data.sample=sample;if(data.capabilities)data.capabilities.saved_corpus=true;data.corpus_revision=sample.revision;transaction.corpusRevision=sample.revision;if(corpusActive()){target.sample_id=sample.sample_id;data.revision=sample.revision;}}
          if(corpusActive()){saved=transaction.pixels.slice();dirty=false;}
          if(typeof corpusUI!=='undefined')await corpusUI.refresh().catch(()=>{});
        }
      }catch(error){transaction.results[destination]={ok:false,error:error.message};}
    }
    const ok=transaction.destinations.every(destination=>transaction.results[destination]?.ok);
    const results={...transaction.results};
    text('mask-save-status',transaction.destinations.map(destination=>`${destination==='workspace'?'Workspace mask':'Corpus label'}: ${results[destination]?.ok?'saved':results[destination]?.error||'not saved'}`).join(' · '));
    if(ok){frozenSave=null;history=[];status(corpusActive()?'Corpus label saved.':transaction.destinations.length===1&&transaction.destinations[0]==='corpus'?'Corpus label saved; the workspace draft is separate.':'Mask saved. Refit creates pose candidates for review.');}
    else status('Some selected saves failed. Retry Save to complete only the failed destination.');
    busy=false;controls();draw();return {ok,destinations:results};
  }
  async function saveRefit() {
    if(!ready()||corpusActive()||!state.region)return false;
    const region={...state.region},source=currentSourceKey(),frame=currentFrameIndex();
    const result=await save();
    if(!result.ok||!result.destinations.workspace?.ok||source!==currentSourceKey())return false;
    if(frame!==currentFrameIndex()||state.region?.first!==region.first||state.region?.last!==region.last){status('Mask saved. Selection changed during saving; choose the intended range and prepare its refit.');return false;}
    status('Mask saved; current pose is stale. Run a fit, then compare candidates before accepting.');
    return refit(true);
  }
  async function saveNext() {const result=await save();return result.ok?await nav()?.next():false;}
  async function removeOverride() {
    if(!ready()||corpusActive()||dirty||!data.has_override)return {ok:false};busy=true;controls();
    try {const result=await api(editsUrl('mask')+`?frame=${data.frame}&revision=${encodeURIComponent(data.revision||'')}`,{method:'DELETE'});if(typeof applyEditResponse==='function')await applyEditResponse(result);busy=false;invalidate();await onFrame(true);status('Override removed. The automatic mask is restored; undo this saved operation in History.');return {ok:true};}
    catch(error){status(error.message);return {ok:false};}finally{busy=false;controls();}
  }
  async function deleteCorpus() {
    if(!ready()||!data.sample)return false;
    if(!await requestLeave({preserveTarget:true}))return false;
    if(!window.confirm('Delete this saved corpus label?'))return false;
    busy=true;controls();
    try {await api(`/api/corpus/labels/${encodeURIComponent(data.sample.sample_id)}`,{method:'DELETE'});data.sample=null;data.corpus_revision=0;if(data.capabilities)data.capabilities.saved_corpus=false;if(target.sample_id)delete target.sample_id;status(corpusActive()?'Saved corpus label deleted. The current pixels remain as a draft.':'Independent corpus label deleted.');if(corpusActive())dirty=true;await corpusUI.refresh();return true;}
    catch(error){status(error.message);return false;}finally{busy=false;controls();}
  }
  function sourceName(name) {return name==='saved'?(value('mask-saved-source',corpusActive()?'corpus':'override')==='corpus'?'saved_corpus':'saved_workspace'):name;}
  let probability=null;
  function updateThreshold() {
    const threshold=Number(value('mask-threshold','.5'));text('mask-threshold-value',threshold.toFixed(2));
    if(probability&&proposalName==='network'){proposal=probability.map(v=>v>=threshold*255?255:0);refreshOverlay();}
  }
  async function selectProposal(name) {
    if(!ready())return false;const source=sourceName(name);if(data.capabilities?.[source]===false){text('mask-proposal-status','This proposal is unavailable for the current target.');return false;}
    const token=generation,request=++proposalRequest;proposalName=name;proposal=null;probability=null;controls();text('mask-proposal-status',`Loading ${source} preview…`);
    try {
      const response=await post('/api/labeling/proposals',{target,source,request_id:request});
      const [mask,prob]=await Promise.all([decodeGray(response.mask),decodeGray(response.probability)]);
      if(token!==generation||request!==proposalRequest)return false;
      if(!mask&&!prob)throw new Error('The proposal has no mask.');
      if((mask||prob).length!==pixels.length)throw new Error('Proposal dimensions do not match this frame.');
      probability=prob;proposal=normalize(mask);if(probability)updateThreshold();else refreshOverlay();
      text('mask-proposal-status',`${source.replaceAll('_',' ')} preview · Apply changes the draft.`);return true;
    }catch(error){if(token===generation&&request===proposalRequest){proposal=null;probability=null;text('mask-proposal-status',error.message);controls();}return false;}
  }
  function applyProposal(event) {
    if(!ready()||!proposal)return false;
    const mode=event?.shiftKey?'union':event?.altKey?'intersect':event?.ctrlKey||event?.metaKey?'subtract':value('mask-combine','replace');
    pushUndo();
    for(let i=0;i<pixels.length;i++){const p=proposal[i]===255,w=pixels[i]===255,other=pixels[i]===127?127:0;
      pixels[i]=mode==='union'?(p||w?255:other):mode==='intersect'?(p&&w?255:other):mode==='subtract'?(w&&!p?255:other):proposal[i];}
    mutated();return true;
  }
  async function refine(method) {
    if(!ready()||refining)return false;
    const token=generation,revision=draftGeneration,request=++refineRequest,mask=encode();refining=true;controls();text('mask-refine-status',`Running ${method.replaceAll('_',' ')}…`);
    try {
      const response=await post('/api/labeling/refine',{target:structuredClone(target),mask,method,request_id:request,draft_generation:revision});const labels=normalize(await decodeGray(response.mask));
      if(token!==generation||request!==refineRequest)return false;
      if(revision!==draftGeneration){text('mask-refine-status','The draft changed while refinement ran; its result was discarded.');return false;}
      if(!labels||labels.length!==pixels.length)throw new Error('Refinement dimensions do not match this frame.');
      pushUndo();pixels=labels;mutated();text('mask-refine-status',`${method.replaceAll('_',' ')} applied${response.info?' · '+JSON.stringify(response.info):''}`);return true;
    }catch(error){if(token===generation)text('mask-refine-status',error.message);return false;}
    finally{if(token===generation&&request===refineRequest){refining=false;controls();}}
  }
  function setBrush(value) {if(!ready())return false;brush=typeof value==='string'?({worm:255,background:0,ignore:127}[value]??Number(value)):value;controls();draw();return true;}
  function resizeBrush(delta) {if(!ready())return false;diameter=Math.max(1,Math.min(100,diameter+delta));if(node('mask-size'))node('mask-size').value=diameter;text('mask-size-value',diameter+' px');draw();return true;}
  function dab(x,y) {const r=diameter/2;for(let yy=Math.max(0,Math.floor(y-r));yy<Math.min(data.height,Math.ceil(y+r));yy++)for(let xx=Math.max(0,Math.floor(x-r));xx<Math.min(data.width,Math.ceil(x+r));xx++)if((xx+.5-x)**2+(yy+.5-y)**2<=r*r)pixels[yy*data.width+xx]=brush;}
  function stroke(a,b) {const n=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)/Math.max(1,diameter/4)));for(let i=0;i<=n;i++)dab(a.x+(b.x-a.x)*i/n,a.y+(b.y-a.y)*i/n);mutated();}
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
    if(enabled()&&cursor){ctx.save();ctx.beginPath();ctx.arc(cursor.x,cursor.y,diameter/2,0,Math.PI*2);ctx.strokeStyle=brush===255?'#ff4fa3':brush===127?'#ffd23c':'#57d68d';ctx.lineWidth=1.5/state.view.scale;ctx.stroke();ctx.restore();}
  }
  function drawCorpus() {
    if(!corpusActive()||!data)return false;const ratio=window.devicePixelRatio||1,v=state.view;
    ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.setTransform(ratio*v.scale,0,0,ratio*v.scale,ratio*v.tx,ratio*v.ty);ctx.drawImage(state.showRaw&&data._raw?data._raw:data._image,0,0);drawLayer();text('caption',`Corpus: ${target.recording||target.source_path||target.sample_id} · frame ${data.frame} · independent label`);return true;
  }
  function rememberWorkspace() {if(workspaceContext)return;workspaceContext={row:state.row,view:{...state.view},region:state.region?structuredClone(state.region):null,timeline:state.timeline?{...state.timeline}:null,showRaw:state.showRaw,compareName:state.compareName,compareKind:state.compareKind,navigation:nav()?.captureContext()};}
  async function openTarget(wanted,options={}) {
    if(!options.guarded&&!await requestLeave({preserveTarget:true}))return false;
    if(wanted.workspace) {
      if(!isWorkspace()||wanted.workspace!==state.runName){status('Open that correction workspace before editing its mask.');return false;}
      const row=state.run.series.frame_index.indexOf(wanted.frame);if(row<0){status('That frame is outside this workspace.');return false;}
      suppressVisit=true;try{invalidate();await showRow(row,{keepView:true,immediate:true,skipMaskGuard:true});await onFrame(true);}finally{suppressVisit=false;}if(!pixels)return false;
    } else {
      const enteringCorpus=!corpusActive(),previousContext=workspaceContext;rememberWorkspace();if(state.playing)togglePlay();const token=++generation;busy=true;controls();
      const progress=typeof beginPreviewTask==='function'?beginPreviewTask('Preparing label preview'):null;
      try {
        if(progress&&wanted.recording)await prepareRecording({path:wanted.recording,...(wanted.dataset?{dataset:wanted.dataset}:{})},progress);
        progress?.update('preview','Reading corrected frame and loading label masks…');
        const payload=await post('/api/labeling/frame',{target:wanted});if(!await install(payload,token,wanted)){progress?.finish();return false;}key=null;if(enteringCorpus)nav()?.beginCorpus({preserveNavigation:!!options.navigation});
        state.view={scale:1,tx:0,ty:0};const rect=canvas.getBoundingClientRect();state.view.scale=Math.min(rect.width/data.width,rect.height/data.height)*.95;
      }catch(error){progress?.finish(error);if(enteringCorpus)workspaceContext=previousContext;status(error.message);return false;}finally{busy=false;controls();}
      progress?.finish();
    }
    if(node('mask-enable'))node('mask-enable').checked=true;active=true;if(typeof showTask==='function')showTask('paint');else if(typeof showTab==='function')showTab('paint');
    if(options.recordVisit!==false)nav()?.record(target);draw();return true;
  }
  async function openCorpus(id) {const opened=await openTarget({sample_id:id});if(opened){if(node('mask-next-mode'))node('mask-next-mode').value='browse';if(node('mask-next-pool'))node('mask-next-pool').value='recordings';if(node('mask-saved-source'))node('mask-saved-source').value='corpus';controls();}return opened;}
  async function returnToWorkspace() {
    if(!await requestLeave({preserveTarget:true}))return false;const context=workspaceContext;invalidate();workspaceContext=null;
    if(context){nav()?.restoreContext(context.navigation);state.view={...context.view};state.region=context.region;if(context.timeline)state.timeline={...context.timeline};state.showRaw=context.showRaw;await showRow(context.row,{keepView:true,immediate:true,skipMaskGuard:true});state.view={...context.view};}
    else if(state.run)await showRow(state.row,{keepView:true,immediate:true,skipMaskGuard:true});
    await onFrame(true);draw();return true;
  }
  async function refit(selection) {if(!beforeMutation())return false;if(typeof prepareMaskRefit==='function')return prepareMaskRefit(selection?'selection':'current');status('Rerun controls are not available.');return false;}
  async function ensureSavedForRerun() {
    if(corpusActive()){status('Return to the workspace before rerunning its poses.');return false;}
    if(busy||pendingEdit())return false;if(!dirty&&!frozenSave)return true;
    const choice=await new Promise(resolve=>{const dialog=document.createElement('dialog'),message=document.createElement('p');message.textContent='Rerun needs a saved mask. Save this draft and use it?';dialog.append(message);for(const [answer,label] of [['save','Save and use mask'],['paint','Return to Paint']]){const button=document.createElement('button');button.textContent=label;button.onclick=()=>{dialog.close();dialog.remove();resolve(answer);};dialog.append(button);}dialog.addEventListener('cancel',event=>{event.preventDefault();dialog.close();dialog.remove();resolve('paint');},{once:true});document.body.append(dialog);dialog.showModal();});
    if(choice==='save')return (await save()).ok;if(typeof showTask==='function')showTask('paint');else showTab('paint');return false;
  }
  function cycleOpacity() {const current=displaySettings().alpha;setDisplay({alpha:current>.4?.2:current>0?0:.45});return displaySettings().alpha;}
  function setActive(value) {active=!!value;if(active&&state.playing)togglePlay();painting=false;cursor=null;controls();draw();}
  function init() {
    if(node('mask-save')&&!node('mask-save-refit')){const button=document.createElement('button');button.id='mask-save-refit';button.textContent='Save & refit selected range';button.title='Save this frame’s mask, then configure Run for the selected range. Candidates require explicit acceptance.';node('mask-save').parentElement.append(button);}
    if(node('mask-save-refit')){const scope=document.createElement('p');scope.id='mask-refit-target';scope.className='note';node('mask-save-refit').parentElement.after(scope);}
    listen('mask-save-refit','click',saveRefit);
    window.addEventListener('workflow:selection',controls);
    listen('mask-corpus-close','click',returnToWorkspace);listen('mask-enable','change',()=>{if(enabled()&&state.playing)togglePlay();onFrame();draw();});
    for(const button of document.querySelectorAll('[data-mask-brush]'))button.onclick=()=>setBrush(Number(button.dataset.maskBrush));
    for(const button of document.querySelectorAll('[data-mask-proposal]'))button.onclick=async event=>{if(await selectProposal(button.dataset.maskProposal)){if(event.shiftKey||event.altKey||event.ctrlKey||event.metaKey)applyProposal(event);}};
    for(const button of document.querySelectorAll('[data-mask-refine]'))button.onclick=()=>refine(button.dataset.maskRefine);
    listen('mask-size','input',event=>{diameter=Math.max(1,Math.min(100,Number(event.target.value)));text('mask-size-value',diameter+' px');draw();});
    listen('mask-visible','change',event=>setDisplay({on:event.target.checked}));
    for(const id of ['mask-predicted','mask-proposal-preview'])listen(id,'change',draw);
    listen('mask-opacity','change',event=>setDisplay({alpha:Number(event.target.value)/100}));
    listen('mask-save','click',()=>save());listen('mask-save-next','click',saveNext);listen('mask-clear','click',removeOverride);listen('mask-clear-draft','click',clearDraft);
    listen('mask-stroke-undo','click',undoStroke);listen('mask-discard','click',discardDraft);listen('mask-to-corpus','click',()=>save({corpusOnly:true}));listen('mask-delete-corpus','click',deleteCorpus);
    listen('mask-refit-frame','click',()=>refit(false));listen('mask-refit-stretch','click',()=>refit(true));listen('mask-also-corpus','change',controls);
    listen('mask-threshold','input',updateThreshold);listen('mask-saved-source','change',()=>{if(proposalName==='saved')selectProposal('saved');});listen('mask-apply-proposal','click',applyProposal);
    canvas.addEventListener('pointerdown',paint,true);
    canvas.addEventListener('pointermove',event=>{cursor=toImage(event.clientX,event.clientY);if(painting){event.stopImmediatePropagation();stroke(last,cursor);last=cursor;}else if(enabled())draw();},true);
    for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,event=>{if(painting){painting=false;last=null;event.stopImmediatePropagation();controls();draw();}},true);
    canvas.addEventListener('pointerleave',()=>{cursor=null;draw();});canvas.addEventListener('contextmenu',event=>{if(enabled())event.preventDefault();});
    window.addEventListener('beforeunload',event=>{if(dirty||busy||frozenSave){event.preventDefault();event.returnValue='';}});nav()?.init();controls();
  }
  return {dimensions:()=>corpusActive()&&data?data:null,init,onFrame,invalidate,leave,requestLeave,beforeMutation,canMutate:()=>!dirty&&!busy&&!pendingEdit()&&!corpusActive()&&!frozenSave,ensureSavedForRerun,undoStroke,draw:drawLayer,drawCorpus,openCorpus,openTarget,returnToWorkspace,isDirty:()=>dirty||!!frozenSave,corpusActive,getTarget:()=>target?structuredClone(target):null,save,saveNext,setBrush,resizeBrush,selectProposal,applyProposal,refine,clearDraft,discardDraft,cycleOpacity,setActive,setDisplay,syncDisplayControls,displayAvailable,next:()=>nav()?.next(),previous:()=>nav()?.previous(),getNavigationPool:()=>nav()?.pool(),setNavigationPool:pool=>nav()?.setPool(pool),refreshControls:controls};
})();
