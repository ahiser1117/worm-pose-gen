// PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_EXECUTABLE=/path/to/chromium node tests/browser/task_labels_review.cjs
// Controller acceptance in a real DOM; API fixtures deliberately use stepped frames
// and same-basename recordings, so accidental row/frame or source mixing fails.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});
  try {
    const page=await browser.newPage();
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.setContent('<div id="root"></div>');
    await page.evaluate(()=>{
      window.$=selector=>document.querySelector(selector);
      const inputIds=['region-first','region-last','region-anchor-before','region-anchor-after','rerun-current-before','rerun-current-after','corpus-filter','corpus-new-frame'];
      const selectIds=['rerun-scope','corpus-source-filter','corpus-split-filter','corpus-recording-filter','corpus-new-recording','seg-checkpoint'];
      const otherIds=['region-info','region-target','region-run','region-run-note','rerun-scope-info','corpus-list','corpus-train','checkpoint-select','checkpoint-status','corpus-train-status','corpus-counts','corpus-new-open','corpus-new-status','corpus-refresh','checkpoint-refresh','corpus-close','corpus-browse','training-add-labels','training-jobs','training-counts','training-readiness','training-base-model'];
      for(const id of [...inputIds,...selectIds,...otherIds]){const node=document.createElement(inputIds.includes(id)?'input':selectIds.includes(id)?'select':'div');node.id=id;document.body.append(node);}
      for(const scope of ['current','selection','workspace'])$('#rerun-scope').add(new Option(scope,scope));
      window.state={run:{series:{frame_index:[10,20,35,50,80]},selected_checkpoint:null},runName:'test',sourceKind:'workspace',row:2,region:{first:0,last:4,anchor_before:null,anchor_after:null},algorithms:[{id:'independent_multistart',parameters:[]}],algorithm:'independent_multistart',regionParams:{},recordings:[{path:'/pool/a/clip.h5',dataset:'/img_nir',frames:100,readable:true},{path:'/pool/b/clip.h5',dataset:'/frames',frames:100,readable:true}],stageValues:{},stages:[],frameCache:new Map(),candidateDetails:new Map(),candidateSets:[],shownSets:[null,null]};
      window.isWorkspace=()=>true;window.serverIsApp=()=>true;window.currentSourceKey=()=>`workspace:${state.runName}`;
      window.setStatus=(message,kind)=>window.status={message,kind};window.setLoading=()=>{};window.drawCharts=()=>{};window.editsUrl=(endpoint,q)=>`/workspace/${endpoint}${q?'?'+q:''}`;
      window.showTab=name=>window.activeTab=name;window.loadJobs=async()=>{};
      window.submitted=[];window.post=async(url,body)=>{submitted.push({url,body});return{id:'job'};};
      window.maskEditor={beforeMutation:()=>true,requestLeave:async()=>true,openTarget:async target=>{window.openedTarget=target;return true;},setNavigationPool:pool=>window.openedPool=pool,returnToWorkspace:()=>{window.returned=true;},openCorpus:id=>{window.openedSample=id;}};
      window.api=async()=>({anchor_before:1,anchor_after:3});
    });
    await page.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/regions.js','utf8')});
    await page.evaluate(()=>{
      window.renderAlgorithmSelect=()=>{};window.renderAlgorithmForm=()=>{};window.dismissCandidateJobs=()=>{};
      writeRegionForm();setRerunScope('current');
    });
    await page.evaluate(async()=>{await prepareMaskRefit('current');await runRegion();});
    assert.deepEqual(await page.evaluate(()=>state.region),{first:0,last:4,anchor_before:null,anchor_after:null});
    assert.deepEqual(await page.evaluate(()=>[submitted[0].body.first,submitted[0].body.last,submitted[0].body.anchor_before,submitted[0].body.anchor_after]),[35,35,20,50]);
    await page.evaluate(async()=>{setRerunScope('selection');await runRegion();});
    assert.deepEqual(await page.evaluate(()=>[submitted[1].body.first,submitted[1].body.last]),[10,80]);
    await page.evaluate(async()=>{setRerunScope('workspace');await runRegion();});
    assert.equal(await page.evaluate(()=>submitted.length),2,'whole-workspace scope must not submit a regional job');
    await page.evaluate(async()=>{
      setRerunScope('current');window.resolveSave=null;maskEditor.ensureSavedForRerun=()=>new Promise(resolve=>window.resolveSave=resolve);
      const pending=runRegion();state.row=3;resolveSave(true);await pending;delete maskEditor.ensureSavedForRerun;
    });
    assert.equal(await page.evaluate(()=>submitted.length),2,'changed target during Save and use must not submit an old scope on a new target');
    await page.evaluate(()=>{state.row=2;});
    // A slow anchor request for frame 35 cannot attach to the next playhead.
    await page.evaluate(async()=>{setRerunScope('current');window.resolveAnchor=null;api=()=>new Promise(resolve=>window.resolveAnchor=resolve);window.anchorPending=proposeAnchors();state.row=3;resolveAnchor({anchor_before:0,anchor_after:4});await anchorPending;refreshRerunScope();});
    assert.deepEqual(await page.evaluate(()=>[effectiveRerunRegion().first,effectiveRerunRegion().anchor_before]),[3,null]);
    await page.evaluate(()=>{state.candidateSets=[{id:'stale',stale:true,rows:[0,4]}];});
    await page.evaluate(()=>acceptCandidateSet('stale'));
    assert.equal(await page.evaluate(()=>submitted.length),2,'stale candidates must never submit acceptance');
    await page.evaluate(()=>{
      window.corpusRequests=[];
      window.api=async url=>{
        if(url==='/api/checkpoints')return{checkpoints:[]};
        corpusRequests.push(url);
        return{samples:[{sample_id:'sample-b',source_path:'/pool/b/clip.h5',dataset_path:'/frames',frame_index:35,split:'train',revision:7,label_source:'manual:corpus'}],counts:{train:4,val:1,test:0},facets:{sources:['manual:corpus'],splits:['train','val','test'],recordings:[{id:'canonical-b',path:'/pool/b/clip.h5',dataset:'/frames'}]}};
      };
    });
    await page.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/corpus.js','utf8')});
    await page.evaluate(async()=>{corpusUI.init();await corpusUI.refresh();$('#corpus-recording-filter').value='canonical-b';$('#corpus-split-filter').value='train';await corpusUI.refresh();});
    assert.equal(await page.evaluate(()=>new URL(corpusRequests.at(-1),'http://local').searchParams.get('recording')),'canonical-b');
    assert.match(await page.locator('#corpus-list').textContent(),/\/pool\/b\/clip.h5/);
    assert.match(await page.locator('#corpus-list').textContent(),/revision 7/);
    await page.evaluate(async()=>{await document.getElementById('corpus-browse').onclick();});
    assert.equal(await page.evaluate(()=>openedSample),'sample-b','Browse filtered labels opens the first matching saved target');
    await page.evaluate(async()=>{$('#corpus-new-recording').value=JSON.stringify(['/pool/b/clip.h5','/frames']);$('#corpus-new-frame').value='35';await corpusUI.openNewLabel();});
    assert.deepEqual(await page.evaluate(()=>openedTarget),{recording:'/pool/b/clip.h5',dataset:'/frames',frame:35});
    assert.deepEqual(await page.evaluate(()=>openedPool),{recordings:[{recording:'/pool/b/clip.h5',dataset:'/frames'}]});
    assert.equal(await page.evaluate(()=>state.runName),'test');
    await page.evaluate(()=>{document.getElementById('corpus-close').onclick();});
    assert.equal(await page.evaluate(()=>returned),true);
    // Later filter response must win even when an older request resolves last.
    await page.evaluate(async()=>{
      window.pendingCorpus=[];api=url=>url==='/api/checkpoints'?Promise.resolve({checkpoints:[]}):new Promise(resolve=>pendingCorpus.push(resolve));
      const old=corpusUI.refresh();$('#corpus-filter').value='latest';const newer=corpusUI.refresh();
      pendingCorpus[1]({samples:[],counts:{train:0,val:0,test:0},root:'newer'});await newer;
      pendingCorpus[0]({samples:[],counts:{train:0,val:0,test:0},root:'obsolete'});await old;
    });
    assert.match(await page.locator('#corpus-counts').textContent(),/newer/);
    assert.doesNotMatch(await page.locator('#corpus-counts').textContent(),/obsolete/);
    await page.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/edits.js','utf8')});
    await page.evaluate(async()=>{
      const info=document.createElement('div');info.id='run-info';document.body.append(info);
      state.segments=new Map();state.run.provenance=null;
      window.invalidations=0;maskEditor.invalidate=()=>invalidations++;
      window.renderEdits=()=>{};window.populateWorst=()=>{};window.renderProvenanceLegend=()=>{};window.describeSource=()=>'';
      window.frameOptions=[];window.showRow=async(row,options)=>frameOptions.push(options);
      await applyEditResponse({edits:[]},{preserveMaskDraft:true});
    });
    assert.equal(await page.evaluate(()=>invalidations),0,'workspace save must retain the Paint transaction for its corpus destination');
    assert.equal(await page.evaluate(()=>frameOptions[0].skipMaskGuard),true);
    await page.evaluate(async()=>{await applyEditResponse({edits:[]});});
    assert.equal(await page.evaluate(()=>invalidations),1,'ordinary saved pose edits invalidate the mask cache');
    assert.deepEqual(errors,[]);
    console.log('PASS: exact rerun scopes, preserved selection, stale anchors/candidates, canonical corpus filters/new labels, stale filter responses, corpus return adapter');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
