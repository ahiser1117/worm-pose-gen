// PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_EXECUTABLE=/path/to/chromium node tests/browser/task_shortcuts.cjs
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});
  try {
    const page=await browser.newPage();const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.setContent('<canvas id="canvas" tabindex="0"></canvas><button id="toggle-raw">Raw</button><input id="note-comment"><div id="drawer-shortcuts"></div><input id="text"><select id="select"><option>One</option></select><textarea id="textarea"></textarea><div id="editable" contenteditable="true"><span id="editable-child">Type</span></div><button id="native-button">Native</button><a id="native-link" href="#">Link</a><dialog id="dialog">Choice</dialog>');
    await page.evaluate(()=>{
      window.$=selector=>document.querySelector(selector);window.calls=[];
      window.state={screen:'workspace',run:{},activeTask:'inspect'};
      window.currentTab=()=>state.activeTask;window.dirty=false;window.corpus=false;
      const record=name=>(...args)=>calls.push([name,...args]);
      window.togglePlay=record('play');window.step=record('step');window.jump=record('jump');window.toggleLayer=record('layer');window.fitView=record('fit');window.computeStarts=record('starts');window.showTab=record('tab');window.setStatus=record('error');window.undoEdit=record('saved-undo');
      $('#toggle-raw').onclick=record('raw');
      window.maskEditor={isDirty:()=>dirty,corpusActive:()=>corpus};
      for(const action of ['cycleOpacity','setBrush','resizeBrush','selectProposal','applyProposal','refine','next','previous','save','saveNext','undoStroke'])maskEditor[action]=record(action);
      window.sendKey=async(spec,target='canvas')=>{calls=[];const event=new KeyboardEvent('keydown',{bubbles:true,cancelable:true,...spec});document.getElementById(target).dispatchEvent(event);await Promise.resolve();await Promise.resolve();return{calls:[...calls],prevented:event.defaultPrevented};};
    });
    await page.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/shortcuts.js','utf8')});
    await page.evaluate(()=>shortcuts.init());
    const entries=await page.evaluate(()=>shortcuts.entries.map(entry=>({chord:entry.chord,label:entry.label})));
    const eventFor=chord=>{const pieces=chord.split('+'),key=pieces.pop();return{key:key==='space'?' ':key==='enter'?'Enter':key.startsWith('arrow')?'Arrow'+key.slice(5,6).toUpperCase()+key.slice(6):key,ctrlKey:pieces.includes('ctrl'),metaKey:pieces.includes('meta'),altKey:pieces.includes('alt'),shiftKey:pieces.includes('shift')};};
    const normalized=await page.evaluate(events=>events.map(event=>shortcuts.chord(new KeyboardEvent('keydown',event))),entries.map(entry=>eventFor(entry.chord)));
    assert.equal(new Set(normalized).size,entries.length,'registered chords must be unique after normalization');
    assert.deepEqual(normalized,entries.map(entry=>entry.chord),'registered chords must already use canonical normalization');
    const meaning=new Map();
    for(const task of ['rerun','inspect','paint','review','export']){
      await page.evaluate(task=>{state.activeTask=task;state.screen='workspace';dirty=false;corpus=false;},task);
      for(const entry of entries){
        const result=await page.evaluate(spec=>sendKey(spec),eventFor(entry.chord));
        assert(result.calls.length<=1,`${entry.chord} dispatched multiple actions in ${task}`);
        if(result.calls.length){const actual=JSON.stringify(result.calls[0]);if(meaning.has(entry.chord))assert.equal(actual,meaning.get(entry.chord),`${entry.chord} changed meaning in ${task}`);else meaning.set(entry.chord,actual);}
      }
    }
    assert.equal(meaning.size,entries.length,'every listed shortcut should enable in at least one supported task');
    for(const screen of ['import','open','labels','training']){
      await page.evaluate(screen=>{state.screen=screen;state.activeTask='inspect';},screen);
      for(const entry of entries)assert.deepEqual((await page.evaluate(spec=>sendKey(spec),eventFor(entry.chord))).calls,[],`accelerator ${entry.chord} active on ${screen}`);
    }
    await page.evaluate(()=>{state.screen='workspace';state.activeTask='paint';});
    for(const target of ['text','select','textarea','editable','editable-child']){
      for(const key of ['s','Enter','z','ArrowRight'])assert.deepEqual((await page.evaluate(({key,target})=>sendKey({key},target),{key,target})).calls,[],`typing in ${target} triggered ${key}`);
    }
    for(const target of ['native-button','native-link'])for(const key of [' ','Enter']){
      const result=await page.evaluate(({key,target})=>sendKey({key},target),{key,target});assert.deepEqual(result.calls,[]);assert.equal(result.prevented,false,'native activation must remain untouched');
    }
    await page.evaluate(()=>$('#dialog').showModal());
    assert.deepEqual((await page.evaluate(()=>sendKey({key:'s'}))).calls,[]);
    await page.evaluate(()=>$('#dialog').close());
    for(const key of ['s','Enter','a','h','z'])assert.deepEqual((await page.evaluate(key=>sendKey({key,repeat:true}),key)).calls,[],`repeat editing ${key}`);
    assert.deepEqual((await page.evaluate(()=>sendKey({key:'=',repeat:true}))).calls,[['resizeBrush',1]]);
    assert.deepEqual((await page.evaluate(()=>sendKey({key:'ArrowRight',repeat:true}))).calls,[['step',1]]);
    for(const task of ['rerun','inspect','paint','review','export']){
      await page.evaluate(task=>{state.activeTask=task;maskEditor.undoStroke=()=>{calls.push(['undoStroke']);return false;};},task);
      for(const modifier of ['ctrlKey','metaKey']){
        const result=await page.evaluate(modifier=>sendKey({key:'z',[modifier]:true}),modifier);
        assert.deepEqual(result.calls,task==='paint'?[['undoStroke']]:[],`${modifier}+Z must only undo draft, never saved edit`);
      }
    }
    await page.evaluate(()=>{state.activeTask='paint';dirty=true;});
    assert.deepEqual((await page.evaluate(()=>sendKey({key:' '}))).calls,[],'dirty Paint draft must disable playback');
    assert.deepEqual((await page.evaluate(()=>sendKey({key:'s',isComposing:true}))).calls,[]);
    assert.deepEqual((await page.evaluate(async()=>{document.getElementById('canvas').addEventListener('keydown',event=>event.preventDefault(),{once:true});return sendKey({key:'s'});})).calls,[],'already-handled events must not trigger accelerators');
    assert.deepEqual((await page.evaluate(()=>sendKey({key:'B',shiftKey:true}))).calls,[['setBrush',255]],'plain shifted letters normalize case');
    await page.evaluate(()=>{corpus=true;state.run=null;});
    assert.deepEqual((await page.evaluate(()=>sendKey({key:'b'}))).calls,[['setBrush',255]],'corpus Paint works without an active workspace');
    assert.deepEqual((await page.evaluate(()=>sendKey({key:'ArrowRight'}))).calls,[],'corpus editing must not seek the correction workspace');
    await page.evaluate(()=>{corpus=false;state.run={};});
    // Browser/clipboard chords retain their native behavior, including redo.
    const conflicts=[];
    for(const modifier of ['ctrlKey','metaKey'])for(const key of ['w','t','c','v','s','n','p']){
      const result=await page.evaluate(({key,modifier})=>sendKey({key,[modifier]:true}),{key,modifier});
      if(result.calls.length||result.prevented)conflicts.push({key,modifier,result});
    }
    for(const modifier of ['ctrlKey','metaKey']){
      const result=await page.evaluate(modifier=>sendKey({key:'Z',shiftKey:true,[modifier]:true}),modifier);
      if(result.calls.length||result.prevented)conflicts.push({key:'Shift+Z',modifier,result});
    }
    assert.deepEqual(errors,[]);
    assert.deepEqual(conflicts,[],'modified browser/clipboard shortcuts must not be reassigned');
    console.log(`PASS: ${entries.length} canonical shortcuts across five tasks; focus/dialog/native controls; undo isolation; repeat restrictions; browser modifiers`);
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
