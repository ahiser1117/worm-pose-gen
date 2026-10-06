// Standalone workspace Masks controller regression: no server or segmentation model needed.
// PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_EXECUTABLE=/path/to/chromium node tests/browser/masks_task.cjs
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});
 try {
  const page=await browser.newPage({viewport:{width:1000,height:700}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.setContent('<canvas id="canvas" width="200" height="200" style="position:absolute;left:0;top:0"></canvas><div id="fixture" style="position:absolute;left:300px"></div>');
  await page.evaluate(()=>{
   window.$=selector=>document.querySelector(selector);const box=$('#fixture');
   const inputs={'mask-enable':'checkbox','mask-visible':'checkbox','mask-predicted':'checkbox','mask-proposal-preview':'checkbox','mask-also-corpus':'checkbox','mask-size':'range','mask-threshold':'range'};
   for(const [id,type] of Object.entries(inputs)){const n=document.createElement('input');n.id=id;n.type=type;box.append(n);}
   const selects={'mask-saved-source':['override','corpus'],'mask-combine':['replace','union','intersect','subtract'],'mask-corpus-split':['','train','val','test'],'mask-opacity':['45','20','0']};
   for(const [id,values] of Object.entries(selects)){const n=document.createElement('select');n.id=id;for(const value of values){const option=document.createElement('option');option.value=option.textContent=value;n.append(option);}box.append(n);}
   for(const id of ['mask-size-value','mask-status','mask-save','mask-clear','mask-to-corpus','mask-refit-frame','mask-refit-stretch','mask-stroke-undo','mask-discard','mask-target','mask-clear-draft','mask-threshold-value','mask-apply-proposal','mask-proposal-status','mask-refine-status','mask-save-status','mask-save-refit','caption']){const n=document.createElement('button');n.id=id;n.textContent=id;box.append(n);}
   for(const value of ['255','0','127']){const n=document.createElement('button');n.dataset.maskBrush=value;box.append(n);}
   for(const value of ['network','classical','raw_threshold','saved']){const n=document.createElement('button');n.dataset.maskProposal=value;box.append(n);}
   for(const value of ['fill_holes','largest_component','dilate','erode','mask_fit']){const n=document.createElement('button');n.dataset.maskRefine=value;box.append(n);}
   $('#mask-enable').checked=$('#mask-visible').checked=$('#mask-proposal-preview').checked=true;
   $('#mask-threshold').min='.05';$('#mask-threshold').max='.95';$('#mask-threshold').step='.05';$('#mask-threshold').value='.5';$('#mask-size').value='12';
   window.canvas=$('#canvas');window.ctx=canvas.getContext('2d');window.state={playing:false,view:{scale:2,tx:20,ty:30},row:0,runName:'test',run:{series:{frame_index:[5,10,15]}},region:{first:0,last:1},timeline:{start:0,end:2},showRaw:false,frameCache:new Map()};
   window.serverIsApp=()=>true;window.isWorkspace=()=>true;window.currentSourceKey=()=>'ws:test';window.currentFrameIndex=()=>state.run.series.frame_index[state.row];window.editsUrl=()=>'/mask';window.setStatus=message=>window.message=message;window.togglePlay=()=>{state.playing=false;};window.draw=()=>{};window.showTab=task=>window.task=task;
   window.toImage=(x,y)=>({x:(x-state.view.tx)/state.view.scale,y:(y-state.view.ty)/state.view.scale});
   window.image=document.createElement('canvas');image.width=image.height=64;
   window.loadImage=async value=>value?image:null;
   window.decodeGray=async value=>{
    if(value==null)return null;if(Array.isArray(value))return Uint8Array.from(value);if(value==='zero')return new Uint8Array(4096);
    const img=new Image();img.src=value;await img.decode();const c=document.createElement('canvas');c.width=img.width;c.height=img.height;const ctx=c.getContext('2d');ctx.drawImage(img,0,0);const bytes=ctx.getImageData(0,0,c.width,c.height).data;return Uint8Array.from({length:c.width*c.height},(_,i)=>bytes[i*4]);
   };
   window.revision=0;window.corpusRevision=0;window.calls=[];window.failCorpus=false;window.savedMask='zero';
   window.framePayload=target=>({target:{recording:'/r.h5',dataset:'/img_nir',...target},frame:target.frame??5,width:64,height:64,image:'image',image_raw:'image',mask:savedMask,base_mask:'zero',has_override:revision>0,revision:target.workspace?'r'+revision:corpusRevision,corpus_revision:corpusRevision,sample:corpusRevision?{sample_id:'s1',revision:corpusRevision}:null,capabilities:{network:true,classical:true,raw_threshold:true,saved_workspace:revision>0,saved_corpus:corpusRevision>0}});
   window.post=async(url,body)=>{
    calls.push({url,body:structuredClone(body)});
    if(url==='/api/labeling/frame')return framePayload(body.target);
    if(url==='/api/labeling/proposals'){if(window.pendingProposal)return new Promise(resolve=>window.resolveProposal=resolve);const mask=new Array(4096).fill(0);mask[0]=255;mask[1]=128;return {mask,probability:body.source==='network'?new Array(4096).fill(100):null};}
    if(url==='/api/labeling/refine'){if(window.pendingRefine)return new Promise(resolve=>window.resolveRefine=resolve);return {mask:body.mask,info:{method:body.method}};}
    if(url==='/mask'){if(window.pendingSave)await new Promise(resolve=>window.resolveSave=resolve);revision++;savedMask=body.mask;return {edit:{},series_patch:{}};}
    if(url==='/api/labeling/save'){if(failCorpus)throw Error('corpus write failed');corpusRevision++;return {sample:{sample_id:'s1',revision:corpusRevision}};}
    throw Error('Unexpected route '+url);
   };
   window.api=async url=>framePayload({workspace:'test',frame:currentFrameIndex()});
   window.applyEditResponse=async(payload,options)=>{if(!options?.preserveMaskDraft)throw Error('save must preserve draft transaction');};
   window.corpusUI={refresh:async()=>{}};
   window.requestDraftDecision=async()=>window.decision||'stay';
  });
  for(const file of ['mask_tools.js','masks.js'])await page.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/'+file,'utf8')});
  await page.evaluate(async()=>{maskEditor.init();await maskEditor.onFrame(true);});
  // Transformed strokes; every restored pan gesture leaves pixels untouched.
  for(const gesture of [{button:'middle'},{button:'right'},{mod:'Shift'},{mod:'Alt'}]){
   if(gesture.mod)await page.keyboard.down(gesture.mod);await page.mouse.move(40,50);await page.mouse.down({button:gesture.button||'left'});await page.mouse.move(70,50);await page.mouse.up({button:gesture.button||'left'});if(gesture.mod)await page.keyboard.up(gesture.mod);
   assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false,JSON.stringify(gesture));
  }
  await page.mouse.move(40,50);await page.mouse.down();await page.mouse.move(80,50,{steps:4});await page.mouse.up();
  assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);assert.equal(await page.evaluate(()=>maskEditor.undoStroke()),true);assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false);
  // Proposal preview never edits; apply and undo preserve PNG ignore values.
  await page.evaluate(()=>maskEditor.selectProposal('classical'));assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false);
  await page.evaluate(()=>maskEditor.applyProposal());assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);
  await page.evaluate(async()=>{await maskEditor.save({corpusOnly:true});});
  assert.deepEqual(await page.evaluate(async()=>Array.from((await decodeGray(calls.filter(c=>c.url==='/api/labeling/save').at(-1).body.mask)).slice(0,3))),[255,127,0]);
  await page.evaluate(()=>maskEditor.discardDraft());
  await page.evaluate(async()=>{await maskEditor.selectProposal('network');$('#mask-threshold').value='.75';$('#mask-threshold').dispatchEvent(new Event('input'));});assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false);
  // Combined modes keep ignore where the legacy combine rules preserve it.
  for(const mode of ['replace','union','intersect','subtract']){
   await page.evaluate(async mode=>{await maskEditor.selectProposal('classical');$('#mask-combine').value=mode;maskEditor.applyProposal();},mode);
   assert.equal(await page.evaluate(()=>maskEditor.undoStroke()),true);
  }
  // All refine operations share undo; slow refinement cannot overwrite newer work.
  for(const method of ['fill_holes','largest_component','dilate','erode','mask_fit']){assert.equal(await page.evaluate(method=>maskEditor.refine(method),method),true);assert.equal(await page.evaluate(()=>maskEditor.undoStroke()),true);}
  await page.evaluate(()=>{pendingRefine=true;window.refinePromise=maskEditor.refine('mask_fit');});
  await page.waitForFunction(()=>!!window.resolveRefine);
  await page.evaluate(()=>{maskEditor.clearDraft();resolveRefine({mask:new Array(4096).fill(255)});});
  assert.equal(await page.evaluate(()=>refinePromise),false);await page.evaluate(()=>{pendingRefine=false;maskEditor.discardDraft();});
  // Partial saves freeze bytes and successful destinations; retry does not duplicate workspace writes.
  await page.evaluate(async()=>{await maskEditor.selectProposal('classical');maskEditor.applyProposal();$('#mask-also-corpus').checked=true;failCorpus=true;calls=[];window.result=await maskEditor.save();});
  assert.equal(await page.evaluate(()=>result.ok),false);assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);
  assert.deepEqual(await page.evaluate(()=>calls.filter(c=>['/mask','/api/labeling/save'].includes(c.url)).map(c=>c.url)),['/mask','/api/labeling/save']);
  await page.evaluate(async()=>{failCorpus=false;window.result=await maskEditor.save();});assert.equal(await page.evaluate(()=>result.ok),true);
  assert.equal(await page.evaluate(()=>calls.filter(c=>c.url==='/mask').length),1);assert.equal(await page.evaluate(()=>calls.filter(c=>c.url==='/api/labeling/save').length),2);
  assert.equal(await page.evaluate(()=>calls.filter(c=>c.url==='/api/labeling/save').every(c=>c.body.mask===calls.find(c=>c.url==='/mask').body.mask)),true);
  // Guard supports Stay, Discard and Save; tab changes alone preserve a draft.
  await page.evaluate(()=>{maskEditor.clearDraft();maskEditor.setBrush(255);});
  await page.mouse.move(40,50);await page.mouse.down();await page.mouse.up();
  assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);
  await page.evaluate(()=>maskEditor.setActive(false));assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);await page.evaluate(()=>maskEditor.setActive(true));
  assert.equal(await page.evaluate(()=>maskEditor.requestLeave({preserveTarget:true})),false);
  await page.evaluate(()=>{decision='discard';});assert.equal(await page.evaluate(()=>maskEditor.requestLeave({preserveTarget:true})),true);assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false);
  // A late proposal from the previous frame cannot become the new draft input.
  await page.evaluate(()=>{pendingProposal=true;window.proposalPromise=maskEditor.selectProposal('classical');});await page.waitForFunction(()=>!!window.resolveProposal);
  await page.evaluate(async()=>{state.row=1;await maskEditor.onFrame(true);resolveProposal({mask:new Array(4096).fill(255)});});assert.equal(await page.evaluate(()=>proposalPromise),false);assert.equal(await page.evaluate(()=>maskEditor.applyProposal()),false);
  await page.evaluate(()=>{pendingProposal=false;});
  // Save & refit freezes its intended range: changing selection while saving
  // must preserve the new selection and never route an unintended fit.
  await page.evaluate(()=>{window.prepared=[];window.prepareMaskRefit=async scope=>{prepared.push(scope);return true;};$('#mask-also-corpus').checked=false;window.pendingSave=true;state.region={first:0,last:1};window.beforeView={...state.view};$('#mask-save-refit').click();});
  await page.waitForFunction(()=>!!window.resolveSave);
  assert.equal(await page.evaluate(()=>maskEditor.requestLeave({preserveTarget:true})),false);
  await page.evaluate(()=>{state.region={first:1,last:2};resolveSave();});
  await page.waitForFunction(()=>!document.getElementById('mask-save').disabled);
  assert.deepEqual(await page.evaluate(()=>prepared),[]);
  assert.deepEqual(await page.evaluate(()=>state.region),{first:1,last:2});
  assert.deepEqual(await page.evaluate(()=>state.view),await page.evaluate(()=>beforeView));
  assert.match(await page.locator('#mask-status').textContent(),/Selection changed/);
  await page.evaluate(()=>{window.pendingSave=false;$('#mask-save-refit').click();});
  await page.waitForFunction(()=>prepared.length===1);
  assert.deepEqual(await page.evaluate(()=>prepared),['selection']);
  assert.deepEqual(await page.evaluate(()=>state.region),{first:1,last:2});
  assert.deepEqual(await page.evaluate(()=>state.view),await page.evaluate(()=>beforeView));
  assert.deepEqual(errors,[]);console.log('Masks passed: pan, draft undo, proposals, ignore labels, refinement races, frozen partial saves, leave guards, late proposals and save & refit.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
