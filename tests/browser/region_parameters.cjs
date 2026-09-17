// PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_EXECUTABLE=/path/to/chrome node tests/browser/region_parameters.cjs
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const tracked=JSON.parse(execFileSync('.venv/bin/python',['-c', 'import json; from worm_pose_gen.algorithms import get_algorithm; print(json.dumps(get_algorithm("tracked_head").to_dict()))'],{encoding:'utf8'}));
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});
  try {
    const page=await browser.newPage();
    await page.setContent('<select id="region-algorithm"></select><div id="region-algorithm-help"></div><main id="region-params"></main><button id="region-run"></button>');
    await page.evaluate(()=>{
      window.state={regionParams:{}};
      window.paramInputType=p=>p.type;
      window.writeStorage=(key,value)=>window.saved=JSON.parse(JSON.stringify(value));
      window.setStatus=message=>{throw new Error(message);};
      window.parseParamInput=(kind,input)=>({value:kind==='bool'?input.checked:Number(input.value)});
    });
    await page.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/regions.js','utf8')});
    await page.evaluate(tracked=>{
      state.algorithms=[tracked];state.algorithm=tracked.id;
      state.run={series:{frame_index:[10,20,35]}};state.runName='test';state.row=1;
      window.$=selector=>document.querySelector(selector);
      window.serverIsApp=()=>true;window.currentSourceKey=()=>"workspace:test";
      window.readCurrentAnchors=()=>true;window.dismissCandidateJobs=()=>{};
      window.renderRegionInfo=()=>{};window.loadJobs=async()=>{};window.setStatus=()=>{};
      window.post=async(url,body)=>{window.submitted={url,body};return{id:'job'};};
      renderAlgorithmForm();
    },tracked);
    assert(await page.locator('#region-algorithm-help').isHidden());
    assert.match(await page.locator('#region-algorithm').getAttribute('title'), /Fits forward/);
    const holes=page.getByRole('combobox',{name:'Hole filling'});
    assert.deepEqual(await holes.locator('option').allTextContents(),['Use workspace masks','On — fill narrow holes','Off — preserve holes']);
    assert.equal(await holes.inputValue(),'off');
    assert.equal(await page.getByRole('spinbutton',{name:'Previous pose strength'}).inputValue(),'0.005');
    assert.equal(await page.getByRole('spinbutton',{name:'Head distance scale (px)'}).inputValue(),'6');
    assert(await page.locator('#region-param-help-tracked_head-fill_holes').isHidden());
    assert.match(await holes.locator('..').getAttribute('title'), /resegment unedited frames/);
    await holes.selectOption('workspace');
    assert.deepEqual(await page.evaluate(()=>collectRegionParams('tracked_head')),{fill_holes:'workspace'});
    assert.deepEqual(await page.evaluate(()=>saved.tracked_head),{fill_holes:'workspace'});
    const motion=page.getByRole('spinbutton',{name:'Maximum head movement (px / source frame)'});
    await motion.fill('2.5');await motion.dispatchEvent('change');
    assert.deepEqual(await page.evaluate(()=>collectRegionParams('tracked_head')),{fill_holes:'workspace',max_head_step_px:2.5});
    await page.evaluate(()=>runRegion());
    assert.deepEqual(await page.evaluate(()=>[submitted.url,submitted.body.algorithm,submitted.body.params]),['/api/jobs','tracked_head',{fill_holes:'workspace',max_head_step_px:2.5}]);
    assert.equal(await holes.getAttribute('aria-describedby'),'region-param-help-tracked_head-fill_holes');
    await holes.selectOption('off');
    assert.deepEqual(await page.evaluate(()=>collectRegionParams('tracked_head')),{max_head_step_px:2.5});
    await page.getByRole('button',{name:'defaults',exact:true}).click();
    assert.deepEqual(await page.evaluate(()=>collectRegionParams('tracked_head')),{});
    assert.equal(await motion.inputValue(),'8');
    assert.equal(await holes.inputValue(),'off');
    console.log('Region parameters: updated defaults, hover-only help, saved choices and numeric controls passed.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
