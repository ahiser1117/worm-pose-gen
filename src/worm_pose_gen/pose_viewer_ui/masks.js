"use strict";

// Editing uses the labeler's pixel convention and circular, interpolated brush.
// The draft is separate from all probability/predicted-mask layers.
const maskEditor = (() => {
  let data = null, pixels = null, saved = null, history = [], dirty = false, busy = false;
  let frameTimer=null, frameResolve=null;
  let key = null, generation = 0, corpus = null, painting = false, last = null, cursor = null;
  let brush = 255, diameter = 12, overlay = document.createElement('canvas'), predicted = null;
  const savedEditPending = () => typeof editPending === 'function' && editPending();
  const enabled = () => !!$('#mask-enable').checked;
  const supported = () => serverIsApp() && (isWorkspace() || !!corpus);
  const currentKey = () => `${currentSourceKey()}:${currentFrameIndex()}`;
  const url = () => corpus ? `/api/corpus/labels/${encodeURIComponent(corpus.sample_id)}` : editsUrl('mask');
  function status(message) { $('#mask-status').textContent = message; }
  function controls() {
    const ready = supported() && !!pixels && (corpus || key===currentKey()) && !busy && !savedEditPending();
    for (const id of ['mask-save','mask-clear','mask-to-corpus','mask-refit-frame','mask-refit-stretch','mask-stroke-undo','mask-discard']) $('#'+id).disabled = !ready;
    $('#mask-enable').disabled = !supported() || busy;
    $('#mask-save').textContent = corpus ? 'Save corpus label' : 'Save override';
    $('#mask-save').disabled = !ready || !dirty;
    $('#mask-clear').disabled = !ready || !!corpus || !data.has_override;
    $('#mask-to-corpus').disabled = !ready || !!corpus || dirty || !data.has_override;
    $('#mask-stroke-undo').disabled = !ready || !history.length;
    $('#mask-discard').disabled = !ready || !dirty;
    $('#mask-refit-frame').disabled = $('#mask-refit-stretch').disabled = !ready || !!corpus || dirty;
    document.querySelectorAll('[data-mask-brush]').forEach(b => b.classList.toggle('active', Number(b.dataset.maskBrush) === brush));
  }
  function refreshOverlay() {
    if (!data || !pixels) return;
    overlay.width = data.width; overlay.height = data.height;
    const c = overlay.getContext('2d'), img = c.createImageData(data.width, data.height);
    for (let i=0;i<pixels.length;i++) {
      const v=pixels[i], j=i*4;
      img.data[j]=v===255?255:v===127?255:55; img.data[j+1]=v===255?65:v===127?205:190; img.data[j+2]=v===255?155:v===127?40:140;
      img.data[j+3]=v===0?18:110;
    }
    c.putImageData(img,0,0); controls(); draw();
  }
  function draftStatus() { status(`${corpus ? 'Corpus '+corpusUI.sampleLabel(corpus) : 'Frame '+data.frame}: ${dirty?'unsaved draft':corpus?'saved label':data.has_override?'saved override':'predicted mask (no override)'}${data.stale?' · pose needs refit':''}`); }
  async function install(payload, token) {
    const [decoded,image,raw,base] = await Promise.all([decodeGray(payload.mask || payload.label),loadImage(payload.image),loadImage(payload.image_raw),decodeGray(payload.base_mask)]);
    if (token!==generation) return;
    if (!decoded || !image) throw new Error('Mask or source image is missing');
    data={...payload,width:payload.width||image.width,height:payload.height||image.height,_image:image,_raw:raw};
    predicted=null;
    if(base){predicted=document.createElement('canvas');predicted.width=data.width;predicted.height=data.height;const c=predicted.getContext('2d'),img=c.createImageData(data.width,data.height);for(let i=0;i<base.length;i++)if(base[i]>200){img.data[i*4+1]=220;img.data[i*4+2]=255;img.data[i*4+3]=110;}c.putImageData(img,0,0);}
    pixels=decoded.map(v=>v===128?127:v); saved=pixels.slice(); history=[]; dirty=false;
    refreshOverlay(); draftStatus();
  }
  function onFrame(force=false) {
    clearTimeout(frameTimer);if(frameResolve){frameResolve(false);frameResolve=null;}
    if(force)return loadFrame(true);
    controls();
    if(state.playing)return Promise.resolve(false);
    return new Promise(resolve=>{frameResolve=resolve;frameTimer=setTimeout(()=>{frameResolve=null;loadFrame().then(resolve);},180);});
  }
  async function loadFrame(force=false) {
    if(corpus) return;
    controls();
    if(!supported()) { invalidate(); status('Read-only source: import as a workspace to correct masks.'); return; }
    const wanted=currentKey();
    if(!force && key===wanted) return;
    if(dirty || busy) return;
    key=wanted; pixels=null; data=null; history=[]; controls();
    const token=++generation;
    try { await install(await api(editsUrl('mask',`frame=${currentFrameIndex()}`)),token); }
    catch(error) { if(token===generation){key=null; status(error.message); controls();} }
  }
  function invalidate() { clearTimeout(frameTimer);if(frameResolve){frameResolve(false);frameResolve=null;} generation++; key=null; data=null; pixels=null; saved=null; history=[]; dirty=false; painting=false; controls(); }
  function beforeMutation() {
    if (dirty || busy || savedEditPending() || corpus) { setStatus(corpus?'Return to the workspace before changing its poses.':busy || savedEditPending()?'Wait for the current edit to finish.':'Save or discard the mask draft first.', 'error'); return false; }
    return true;
  }
  function leave() {
    if(dirty || busy) { if(state.playing) togglePlay(); setStatus('Save or discard the mask draft before navigating.', 'error'); return false; }
    if(corpus){corpus=null; $('#corpus-close').hidden=true; $('#mask-corpus-close').hidden=true;}
    invalidate(); return true;
  }
  function encode() {
    const out=document.createElement('canvas'); out.width=data.width; out.height=data.height;
    const c=out.getContext('2d'), img=c.createImageData(out.width,out.height);
    for(let i=0;i<pixels.length;i++){const j=i*4;img.data[j]=img.data[j+1]=img.data[j+2]=pixels[i];img.data[j+3]=255;}
    c.putImageData(img,0,0); return out.toDataURL('image/png');
  }
  async function save(clear=false) {
    if(!supported() || busy || savedEditPending() || !pixels) return;
    busy=true; controls();
    try {
      const body={frame:data.frame,mask:encode(),revision:corpus?corpus.revision:data.revision};
      const payload=clear?await api(url()+`?frame=${data.frame}&revision=${encodeURIComponent(data.revision||'')}`,{method:'DELETE'}):await post(url(),body);
      dirty=false;history=[];
      if(corpus){const detail=await api(url());corpus=detail.sample;await install(detail,++generation);await corpusUI.refresh();}
      else {invalidate(); if(payload.series_patch || payload.edit) await applyEditResponse(payload); else {state.frameCache.clear();await showRow(state.row,{keepView:true,immediate:true});await loadEdits();}}
      setStatus(clear?'Override cleared.':corpus?'Corpus label saved.':'Override saved. Refit to create candidates; save to corpus separately.', 'ok');
    } catch(error){setStatus(error.message,'error');}
    finally{busy=false;controls();if(!corpus && !dirty)onFrame(true);}
  }
  function undoStroke() {
    if(!history.length || busy) return false;
    pixels=history.pop();dirty=pixels.some((v,i)=>v!==saved[i]);refreshOverlay();draftStatus();return true;
  }
  function dab(x,y) {
    const r=diameter/2;
    for(let yy=Math.max(0,Math.floor(y-r));yy<Math.min(data.height,Math.ceil(y+r));yy++)
      for(let xx=Math.max(0,Math.floor(x-r));xx<Math.min(data.width,Math.ceil(x+r));xx++)
        if((xx+.5-x)**2+(yy+.5-y)**2<=r*r)pixels[yy*data.width+xx]=brush;
  }
  function stroke(a,b) {
    const n=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)/Math.max(1,diameter/4)));
    for(let i=0;i<=n;i++)dab(a.x+(b.x-a.x)*i/n,a.y+(b.y-a.y)*i/n);
    dirty=pixels.some((v,i)=>v!==saved[i]);refreshOverlay();draftStatus();
  }
  function paint(event) {
    if(!enabled() || !supported() || !pixels || (!corpus && key!==currentKey()) || busy || savedEditPending() || event.altKey || event.button===1) return;
    event.preventDefault(); event.stopImmediatePropagation();
    if(state.playing)togglePlay();
    painting=true;last=toImage(event.clientX,event.clientY);cursor=last;
    history.push(pixels.slice());if(history.length>30)history.shift();canvas.setPointerCapture(event.pointerId);stroke(last,last);
  }
  function drawLayer() {
    if(!pixels || (!corpus && key!==currentKey()))return;
    if(predicted && $('#mask-predicted').checked)ctx.drawImage(predicted,0,0);
    if($('#mask-visible').checked)ctx.drawImage(overlay,0,0);
    if(enabled() && cursor){ctx.save();ctx.beginPath();ctx.arc(cursor.x,cursor.y,diameter/2,0,Math.PI*2);ctx.strokeStyle=brush===255?'#ff4fa3':brush===127?'#ffd23c':'#57d68d';ctx.lineWidth=1.5/state.view.scale;ctx.stroke();ctx.restore();}
  }
  function drawCorpus() {
    if(!corpus || !data)return false;
    const ratio=window.devicePixelRatio||1,v=state.view;
    ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.setTransform(ratio*v.scale,0,0,ratio*v.scale,ratio*v.tx,ratio*v.ty);
    ctx.drawImage(state.showRaw && data._raw?data._raw:data._image,0,0);drawLayer();
    $('#caption').textContent=`Corpus: ${corpusUI.sampleLabel(corpus)} · independent saved label`;
    return true;
  }
  async function openCorpus(id) {
    if(!leave())return;
    if(state.playing)togglePlay();
    busy=true;controls();
    try {const payload=await api(`/api/corpus/labels/${encodeURIComponent(id)}`);corpus=payload.sample;await install(payload,++generation);$('#corpus-close').hidden=false;$('#mask-corpus-close').hidden=false;$('#mask-enable').checked=true;showTab('view');state.view={scale:1,tx:0,ty:0};const rect=canvas.getBoundingClientRect();state.view.scale=Math.min(rect.width/data.width,rect.height/data.height)*.95;draw();}
    catch(error){corpus=null;setStatus(error.message,'error');}finally{busy=false;controls();}
  }
  async function refit(stretch) {
    if(!beforeMutation())return;
    showTab('pipeline');
    if(stretch)await proposeRegion();else{setRegionRows(state.row,state.row,'corrected mask frame');await proposeAnchors();}
    state.algorithm=stretch?'slow_refit':'independent_multistart';renderAlgorithmSelect();renderAlgorithmForm();renderRegionInfo();
    setStatus('Refit region prepared. Review parameters, Run on region, compare candidates, then explicitly Accept.', 'ok');
  }
  function init() {
    $('#mask-corpus-close').onclick=()=>{if(leave())showRow(state.row,{keepView:true,immediate:true});};
    $('#mask-enable').onchange=()=>{if(enabled() && state.playing)togglePlay();onFrame();draw();};
    document.querySelectorAll('[data-mask-brush]').forEach(b=>b.onclick=()=>{brush=Number(b.dataset.maskBrush);controls();draw();});
    $('#mask-size').oninput=e=>{diameter=Number(e.target.value);$('#mask-size-value').textContent=diameter+' px';draw();};
    $('#mask-visible').onchange=draw;$('#mask-predicted').onchange=draw;
    $('#mask-save').onclick=()=>save();$('#mask-clear').onclick=()=>save(true);
    $('#mask-stroke-undo').onclick=undoStroke;
    $('#mask-discard').onclick=()=>{if(busy)return;pixels=saved.slice();history=[];dirty=false;refreshOverlay();draftStatus();};
    $('#mask-to-corpus').onclick=async()=>{if(!beforeMutation())return;busy=true;controls();try{await post('/api/corpus/labels',{workspace:state.runName,frame:data.frame,...($('#mask-corpus-split').value?{split:$('#mask-corpus-split').value}:{})});setStatus('Independent label saved to corpus.','ok');await corpusUI.refresh();}catch(e){setStatus(e.message,'error');}finally{busy=false;controls();}};
    $('#mask-refit-frame').onclick=()=>refit(false);$('#mask-refit-stretch').onclick=()=>refit(true);
    canvas.addEventListener('pointerdown',paint,true);
    canvas.addEventListener('pointermove',e=>{cursor=toImage(e.clientX,e.clientY);if(painting){e.stopImmediatePropagation();stroke(last,cursor);last=cursor;}else if(enabled())draw();},true);
    for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,e=>{if(painting){painting=false;last=null;e.stopImmediatePropagation();controls();draw();}},true);
    canvas.addEventListener('pointerleave',()=>{cursor=null;draw();});
    canvas.addEventListener('contextmenu',e=>{if(enabled())e.preventDefault();});
    window.addEventListener('beforeunload',e=>{if(dirty||busy){e.preventDefault();e.returnValue='';}});
    controls();
  }
  return {dimensions:()=>corpus&&data?data:null,init,onFrame,invalidate,leave,beforeMutation,undoStroke,draw:drawLayer,drawCorpus,openCorpus,isDirty:()=>dirty,corpusActive:()=>!!corpus};
})();
