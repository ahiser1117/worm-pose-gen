// Run: PLAYWRIGHT_MODULE=/path/to/playwright node tests/browser/mask_editor.cjs
// No server/model required: exercise the real controller and canvas pointer events.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage();
    await page.setContent('<canvas id="canvas" width="200" height="200" style="position:absolute;left:0;top:0"></canvas>');
    await page.evaluate(() => {
      window.$ = selector => document.querySelector(selector);
      for (const id of ['mask-enable','mask-visible','mask-predicted','mask-size','mask-size-value','mask-status','mask-save','mask-clear','mask-to-corpus','mask-refit-frame','mask-refit-stretch','mask-stroke-undo','mask-discard','mask-corpus-close','corpus-close']) {
        const node=document.createElement(id==='mask-enable'||id==='mask-visible'?'input':'button');node.id=id;if(node.tagName==='INPUT')node.type='checkbox';document.body.append(node);
      }
      window.canvas=$('#canvas');window.ctx=canvas.getContext('2d');
      window.state={playing:false,view:{scale:2,tx:20,ty:30}};
      window.serverIsApp=()=>true;window.isWorkspace=()=>true;window.currentSourceKey=()=>'ws:test';window.currentFrameIndex=()=>5;
      window.editsUrl=()=>'/mask';window.setStatus=message=>window.message=message;window.togglePlay=()=>{};
      window.toImage=(x,y)=>({x:(x-state.view.tx)/state.view.scale,y:(y-state.view.ty)/state.view.scale});
      window.draw=()=>{};
      const image=document.createElement('canvas');image.width=image.height=64;
      window.api=async()=>({frame:5,width:64,height:64,image:'image',mask:'mask',has_override:false,revision:'r0'});
      window.loadImage=async()=>image;
      window.decodeGray=async()=>new Uint8Array(64*64);
      window.corpusUI={refresh:async()=>{}};
    });
    await page.addScriptTag({content:fs.readFileSync('src/worm_pose_gen/pose_viewer_ui/masks.js','utf8')});
    await page.evaluate(async()=>{maskEditor.init();await maskEditor.onFrame();$('#mask-enable').checked=true;$('#mask-visible').checked=true;});
    // Image pixel (10,10) maps to screen (40,50). A stroke marks the draft
    // and navigation refuses to discard it, regardless of zoom/translation.
    await page.mouse.move(40,50);await page.mouse.down();await page.mouse.move(80,50,{steps:4});await page.mouse.up();
    assert.equal(await page.evaluate(()=>maskEditor.isDirty()),true);
    assert.equal(await page.evaluate(()=>{ctx.clearRect(0,0,200,200);maskEditor.draw();return ctx.getImageData(10,10,1,1).data[0];}),255);
    assert.equal(await page.evaluate(()=>maskEditor.leave()),false);
    assert.equal(await page.evaluate(()=>maskEditor.undoStroke()),true);
    assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false);
    assert.equal(await page.evaluate(()=>maskEditor.leave()),true);
    await page.evaluate(async()=>{await maskEditor.onFrame();});
    await page.mouse.move(40,50);await page.mouse.down({button:'middle'});await page.mouse.move(80,50);await page.mouse.up({button:'middle'});
    assert.equal(await page.evaluate(()=>maskEditor.isDirty()),false);
    console.log('mask editor: transformed pointer stroke, draft navigation protection, undo and pan passed');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
