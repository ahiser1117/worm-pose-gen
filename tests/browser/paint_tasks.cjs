// Standalone Paint controller regression: no server or segmentation model needed.
// PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_EXECUTABLE=/path/to/chromium node tests/browser/paint_tasks.cjs
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
   const inputs={'mask-enable':'checkbox','mask-visible':'checkbox','mask-predicted':'checkbox','mask-proposal-preview':'checkbox','mask-also-corpus':'checkbox','mask-size':'range','mask-threshold':'range','mask-stride':'number','mask-candidates':'number','mask-manifest':'text'};
   for(const [id,type] of Object.entries(inputs)){const n=document.createElement('input');n.id=id;n.type=type;box.append(n);}
   const selects={'mask-saved-source':['override','corpus'],'mask-combine':['replace','union','intersect','subtract'],'mask-corpus-split':['','train','val','test'],'mask-next-mode':['sequential','queue','uncertain','random','browse'],'mask-next-pool':['workspace','selection','recordings','manifest'],'mask-opacity':['45','20','0']};
   for(const [id,values] of Object.entries(selects)){const n=document.createElement('select');n.id=id;for(const value of values){const option=document.createElement('option');option.value=option.textContent=value;n.append(option);}box.append(n);}
   for(const id of ['mask-size-value','mask-status','mask-save','mask-clear','mask-to-corpus','mask-refit-frame','mask-refit-stretch','mask-stroke-undo','mask-discard','mask-corpus-close','corpus-close','mask-target','mask-clear-draft','mask-threshold-value','mask-apply-proposal','mask-proposal-status','mask-refine-status','mask-save-next','mask-save-status','mask-manifest-load','mask-prev','mask-next','mask-next-status','mask-delete-corpus','caption']){const n=document.createElement('button');n.id=id;n.textContent=id;box.append(n);}
   const recordings=document.createElement('textarea');recordings.id='mask-recordings';box.append(recordings);
   for(const value of ['255','0','127']){const n=document.createElement('button');n.dataset.maskBrush=value;box.append(n);}
   for(const value of ['network','classical','raw_threshold','saved']){const n=document.createElement('button');n.dataset.maskProposal=value;box.append(n);}
   for(const value of ['fill_holes','largest_component','dilate','erode','mask_fit']){const n=document.createElement('button');n.dataset.maskRefine=value;box.append(n);}
   $('#mask-enable').checked=$('#mask-visible').checked=$('#mask-proposal-preview').checked=true;
   $('#mask-threshold').min='.05';$('#mask-threshold').max='.95';$('#mask-threshold').step='.05';$('#mask-threshold').value='.5';$('#mask-size').value='12';$('#mask-stride').value='1';$('#mask-candidates').value='6';
   window.canvas=$('#canvas');window.ctx=canvas.getContext('2d');window.state={playing:false,view:{scale:2,tx:20,ty:30},row:0,runName:'test',run:{series:{frame_index:[5,10,15]}},region:{first:0,last:1},timeline:{start:0,end:2},showRaw:false,frameCache:new Map()};
   window.serverIsApp=()=>true;window.isWorkspace=()=>true;window.currentSourceKey=()=>'ws:test';window.currentFrameIndex=()=>state.run.series.frame_index[state.row];window.editsUrl=()=>'/mask';window.setStatus=message=>window.message=message;window.togglePlay=()=>{state.playing=false;};window.draw=()=>{};window.showTask=task=>window.task=task;
   window.toImage=(x,y)=>({x:(x-state.view.tx)/state.view.scale,y:(y-state.view.ty)/state.view.scale});
   window.image=document.createElement('canvas');image.width=image.height=64;
   window.loadImage=async value=>value?image:null;
   window.decodeGray=async value=>{
    if(value==null)return null;if(Array.isArray(value))return Uint8Array.from(value);if(value==='zero')return new Uint8Array(4096);
    const img=new Image();img.src=value;await img.decode();const c=document.createElement('canvas');c.width=img.width;c.height=img.height;const ctx=c.getContext('2d');ctx.drawImage(img,0,0);const bytes=ctx.getImageData(0,0,c.width,c.height).data;return Uint8Array.from({length:c.width*c.height},(_,i)=>bytes[i*4]);
   };
   window.revision=0;window.corpusRevision=0;window.calls=[];window.failCorpus=false;window.savedMask='zero';window.nextResponse={target:null,exhausted:true,reason:'Complete'};
   window.framePayload=target=>({target:{recording:'/r.h5',dataset:'/img_nir',...target},frame:target.frame??5,width:64,height:64,image:'image',image_raw:'image',mask:savedMask,base_mask:'zero',has_override:revision>0,revision:target.workspace?'r'+revision:corpusRevision,corpus_revision:corpusRevision,sample:corpusRevision?{sample_id:'s1',revision:corpusRevision}:null,capabilities:{network:true,classical:true,raw_threshold:true,saved_workspace:revision>0,saved_corpus:corpusRevision>0}});
   window.post=async(url,body)=>{
    calls.push({url,body:structuredClone(body)});
    if(url==='/api/labeling/frame')return framePayload(body.target);
    if(url==='/api/labeling/proposals'){if(window.pendingProposal)return new Promise(resolve=>window.resolveProposal=resolve);const mask=new Array(4096).fill(0);mask[0]=255;mask[1]=128;return {mask,probability:body.source==='network'?new Array(4096).fill(100):null};}
    if(url==='/api/labeling/refine'){if(window.pendingRefine)return new Promise(resolve=>window.resolveRefine=resolve);return {mask:body.mask,info:{method:body.method}};}
    if(url==='/mask'){if(window.pendingSave)await new Promise(resolve=>window.resolveSave=resolve);revision++;savedMask=body.mask;return {edit:{},series_patch:{}};}
    if(url==='/api/labeling/save'){if(failCorpus)throw Error('corpus write failed');corpusRevision++;return {sample:{sample_id:'s1',revision:corpusRevision}};}
    if(url==='/api/labeling/next'){if(window.pendingNext)return new Promise(resolve=>window.resolveNext=resolve);return structuredClone(nextResponse);}
    if(url==='/api/labeling/manifests')return {id:'manifest1',progress:{remaining:3}};
    throw Error('Unexpected route '+url);
   };
   window.api=async url=>framePayload({workspace:'test',frame:currentFrameIndex()});
   window.applyEditResponse=async(payload,options)=>{if(!options?.preserveMaskDraft)throw Error('save must preserve draft transaction');};
   window.corpusUI={refresh:async()=>{},getFilters:()=>({source:'all',split:'all'}),getPool:()=>({recordings:[]})};
   window.showRow=async(row,options)=>{if(!options?.skipMaskGuard)throw Error('internal navigation must skip duplicate guard');state.row=row;};
   window.requestDraftDecision=async()=>window.decision||'stay';
  });
  for(const file of ['paint_navigation.js','masks.js'])await page.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/'+file,'utf8')});
  await page.evaluate(async()=>{maskEditor.init();await maskEditor.onFrame(true);});
  // Startup --queue catalogs select the loaded manifest once, while controls
  // remain visible and indicate which navigation settings are applicable.
  await page.evaluate(()=>paintNavigation.onCatalog({labeling_manifests:[{id:'startup',path:'/startup.json',name:'Startup queue',entries:[{target:{recording:'/r.h5',dataset:'/img_nir',frame:5}},{target:{recording:'/r.h5',dataset:'/img_nir',frame:10}}],progress:{total:2,labeled:0,remaining:2}}]}));
  assert.equal(await page.locator('#mask-next-mode').inputValue(),'queue');assert.equal(await page.locator('#mask-manifest').inputValue(),'/startup.json');assert.equal(await page.locator('#mask-next-pool').inputValue(),'workspace');
  assert.equal(await page.locator('#mask-stride').isDisabled(),true);assert.equal(await page.locator('#mask-candidates').isDisabled(),true);assert.equal(await page.locator('#mask-recordings').isDisabled(),true);assert.equal(await page.locator('#mask-manifest').isDisabled(),false);assert.equal(await page.locator('#mask-next').isDisabled(),false);assert.equal(await page.locator('#mask-prev').isDisabled(),true);
  assert.equal(await page.locator('#mask-corpus-close').isVisible(),true);assert.equal(await page.locator('#mask-corpus-close').isDisabled(),true);
  await page.evaluate(()=>{$('#mask-next-mode').value='uncertain';$('#mask-next-mode').dispatchEvent(new Event('change'));});assert.equal(await page.locator('#mask-candidates').isDisabled(),false);
  await page.evaluate(()=>paintNavigation.refreshControls({network:false}));assert.equal(await page.locator('#mask-next').isDisabled(),true);assert.equal(await page.locator('#mask-candidates').isDisabled(),true);assert.match(await page.locator('#mask-next-help').textContent(),/checkpoint/);
  await page.evaluate(()=>{paintNavigation.refreshControls({network:true});$('#mask-next-mode').value='sequential';$('#mask-next-mode').dispatchEvent(new Event('change'));paintNavigation.onCatalog({labeling_manifests:[{id:'startup',path:'/startup.json',progress:{total:2,labeled:0,remaining:2}}]});});
  assert.equal(await page.locator('#mask-next-mode').inputValue(),'sequential');assert.equal(await page.locator('#mask-stride').isDisabled(),false);assert.equal(await page.locator('#mask-manifest').isDisabled(),true);
  await page.evaluate(()=>{$('#mask-next-pool').value='recordings';$('#mask-next-pool').dispatchEvent(new Event('change'));});assert.equal(await page.locator('#mask-next').isDisabled(),true);assert.equal(await page.locator('#mask-recordings').isDisabled(),false);
  await page.evaluate(()=>{$('#mask-next-pool').value='workspace';$('#mask-next-pool').dispatchEvent(new Event('change'));});
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
  // Failed Save + next never traverses; successful Save + next honors pool and exhaustion.
  await page.evaluate(async()=>{maskEditor.clearDraft();failCorpus=true;calls=[];await maskEditor.saveNext();});assert.equal(await page.evaluate(()=>calls.some(c=>c.url==='/api/labeling/next')),false);
  await page.evaluate(async()=>{failCorpus=false;await maskEditor.saveNext();});assert.equal(await page.evaluate(()=>calls.filter(c=>c.url==='/api/labeling/next').length),1);
  assert.equal(await page.locator('#mask-next-status').textContent(),'Complete');
  // Guard supports Stay, Discard and Save; tab changes alone preserve a draft.
  await page.evaluate(()=>{maskEditor.clearDraft();maskEditor.setBrush(255);});
  await page.mouse.move(40,50);await page.mouse.down();await page.mouse.up();
  assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);
  await page.evaluate(()=>maskEditor.setActive(false));assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);await page.evaluate(()=>maskEditor.setActive(true));
  assert.equal(await page.evaluate(()=>maskEditor.requestLeave({preserveTarget:true})),false);
  await page.evaluate(()=>{decision='discard';});assert.equal(await page.evaluate(()=>maskEditor.requestLeave({preserveTarget:true})),true);assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false);
  // Explicit workspace selection, visited history, and cross-workspace result rejection.
  await page.evaluate(async()=>{nextResponse={target:{workspace:'test',frame:10},exhausted:false};$('#mask-next-pool').value='selection';calls=[];await maskEditor.next();});
  assert.deepEqual(await page.evaluate(()=>calls.find(c=>c.url==='/api/labeling/next').body.pool),{workspace:'test',frames:[5,10]});assert.equal(await page.evaluate(()=>state.row),1);
  assert.equal(await page.evaluate(()=>maskEditor.previous()),true);assert.equal(await page.evaluate(()=>state.row),0);
  await page.evaluate(async()=>{nextResponse={target:{recording:'/other.h5',dataset:'/img_nir',frame:9},exhausted:false};await maskEditor.next();});assert.equal(await page.evaluate(()=>maskEditor.corpusActive()),false);
  // Corpus uses independent target and restores the workspace view/range exactly.
  await page.evaluate(async()=>{state.view={scale:3,tx:17,ty:19};$('#mask-next-pool').value='selection';await maskEditor.openCorpus('s1');maskEditor.setNavigationPool({recordings:[{recording:'/corpus.h5',dataset:'/img_nir'}]});});assert.equal(await page.evaluate(()=>maskEditor.corpusActive()),true);
  await page.evaluate(()=>{state.view={scale:7,tx:40,ty:50};});assert.equal(await page.evaluate(()=>maskEditor.returnToWorkspace()),true);
  assert.deepEqual(await page.evaluate(()=>state.view),{scale:3,tx:17,ty:19});assert.deepEqual(await page.evaluate(()=>state.region),{first:0,last:1});assert.deepEqual(await page.evaluate(()=>maskEditor.getNavigationPool()),{workspace:'test',frames:[5,10]});
  // Each traversal mode is sent explicitly; loading a manifest retains its identity.
  await page.evaluate(async()=>{nextResponse={target:null,exhausted:true,reason:'Complete'};$('#mask-next-pool').value='workspace';$('#mask-next-mode').value='queue';$('#mask-next-mode').dispatchEvent(new Event('change'));$('#mask-manifest').value='/queue.json';$('#mask-manifest').dispatchEvent(new Event('input'));$('#mask-manifest-load').click();});
  await page.waitForFunction(()=>paintNavigation.manifest()==='manifest1');
  for(const mode of ['queue','uncertain','random','sequential','browse']){
   await page.evaluate(async mode=>{$('#mask-next-mode').value=mode;calls=[];await maskEditor.next();},mode);
   assert.equal(await page.evaluate(()=>calls.find(c=>c.url==='/api/labeling/next').body.mode),mode);
  }
  assert.equal(await page.evaluate(()=>calls.find(c=>c.url==='/api/labeling/next').body.manifest_id),'manifest1');
  // Painting during a slow Next request is guarded again, never silently dropped.
  await page.evaluate(()=>{pendingNext=true;decision='stay';window.nextPromise=maskEditor.next();});await page.waitForFunction(()=>!!window.resolveNext);
  await page.mouse.move(48,50);await page.mouse.down();await page.mouse.up();assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);
  await page.evaluate(()=>resolveNext({target:{workspace:'test',frame:10},exhausted:false}));assert.equal(await page.evaluate(()=>nextPromise),false);assert.equal(await page.evaluate(()=>state.row),0);
  await page.evaluate(()=>{pendingNext=false;decision='save';});assert.equal(await page.evaluate(()=>maskEditor.requestLeave({preserveTarget:true})),true);assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false);
  // A late proposal from the previous frame cannot become the new draft input.
  await page.evaluate(()=>{pendingProposal=true;window.proposalPromise=maskEditor.selectProposal('classical');});await page.waitForFunction(()=>!!window.resolveProposal);
  await page.evaluate(async()=>{await maskEditor.openTarget({workspace:'test',frame:10});resolveProposal({mask:new Array(4096).fill(255)});});assert.equal(await page.evaluate(()=>proposalPromise),false);assert.equal(await page.evaluate(()=>maskEditor.applyProposal()),false);
  await page.evaluate(()=>{pendingProposal=false;});
  // Boot --queue uses the manifest corpus pool even when an ambient workspace
  // has been selected, and it preserves that workspace for an explicit return.
  await page.evaluate(()=>{state.view={scale:4,tx:13,ty:29};calls=[];nextResponse={target:{recording:'/startup.h5',dataset:'/img_nir',frame:2,manifest_id:'startup'},exhausted:false,progress:{remaining:1,total:2,labeled:1}};});
  assert.equal(await page.evaluate(()=>paintNavigation.startStartupQueue()),true);
  assert.deepEqual(await page.evaluate(()=>{const body=calls.find(c=>c.url==='/api/labeling/next').body;return {mode:body.mode,pool:body.pool,current:body.current,manifest_id:body.manifest_id};}),{mode:'queue',pool:{},current:null,manifest_id:'startup'});
  assert.equal(await page.evaluate(()=>maskEditor.corpusActive()),true);assert.equal(await page.evaluate(()=>state.row),1);assert.equal(await page.locator('#mask-next-mode').inputValue(),'queue');assert.equal(await page.locator('#mask-next-pool').inputValue(),'manifest');
  assert.equal(await page.evaluate(()=>calls.some(c=>c.url==='/mask'||c.url==='/api/labeling/save')),false);
  assert.equal(await page.evaluate(()=>paintNavigation.startStartupQueue()),false);
  assert.equal(await page.evaluate(()=>maskEditor.returnToWorkspace()),true);assert.deepEqual(await page.evaluate(()=>state.view),{scale:4,tx:13,ty:29});assert.equal(await page.evaluate(()=>state.row),1);
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
  const startup=await browser.newPage();await startup.setContent('<select id="mask-next-mode"><option value="sequential">Sequential</option><option value="queue">Queue</option><option value="uncertain">Uncertain</option></select><select id="mask-next-pool"><option value="workspace">Workspace</option><option value="manifest">Manifest</option><option value="recordings">Recordings</option></select><input id="mask-manifest"><button id="mask-manifest-load">Load</button><input id="mask-stride" value="1"><input id="mask-candidates" value="6"><textarea id="mask-recordings"></textarea><button id="mask-next">Next</button><button id="mask-prev">Previous</button><div id="mask-next-status"></div>');
  await startup.evaluate(()=>{window.state={info:null,run:null};window.isWorkspace=()=>false;window.serverIsApp=()=>true;window.maskEditor={getTarget:()=>null,isDirty:()=>false};window.showTask=task=>state.task=task;window.post=async()=>({target:null,blocked:true,reason:'Missing source /a.h5'});});
  await startup.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/paint_navigation.js','utf8')});
  await startup.evaluate(()=>{paintNavigation.init();paintNavigation.onCatalog({labeling_manifests:[{id:'initial',path:'/initial.json',name:'Initial queue',entries:[{target:{recording:'/a.h5',dataset:'/img_nir',frame:0}}],progress:{remaining:1,total:1,labeled:0}}]});});
  assert.equal(await startup.locator('#mask-next-pool').inputValue(),'manifest');assert.equal(await startup.locator('#mask-next-mode').inputValue(),'queue');assert.equal(await startup.locator('#mask-next').isDisabled(),false);assert.equal(await startup.locator('#mask-prev').isDisabled(),true);
  await startup.evaluate(()=>paintNavigation.onCatalog({labeling_manifests:[{id:'initial',path:'/initial.json',entries:[],progress:{remaining:0,total:0,labeled:0}}]}));assert.equal(await startup.locator('#mask-next').isDisabled(),true);assert.match(await startup.locator('#mask-next-help').textContent(),/complete/);assert.equal(await startup.evaluate(()=>paintNavigation.startStartupQueue()),false);assert.equal(await startup.evaluate(()=>state.task),'paint');assert.match(await startup.locator('#mask-next-status').textContent(),/Missing source/);await startup.close();
  assert.deepEqual(errors,[]);console.log('Paint passed: pan, draft undo, proposals, ignore labels, refinement races, frozen partial saves, save/next guards, declared pool, visited history and corpus return.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
