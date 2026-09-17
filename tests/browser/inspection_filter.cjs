// Isolated controller regression: no recordings, model jobs, or production writes.
const assert = require('node:assert/strict');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('http://inspection.test/**', route => route.fulfill({contentType:'text/html', body:'<div id="tab-inspect"></div><input id="loop-stretch" type="checkbox">'}));
    const mount = async () => {
      await page.goto('http://inspection.test/');
      await page.evaluate(() => {
        window.state = {runName:'fixture',run:{},region:null,row:0,playing:false};
        const segment = (first,last) => ({first,last,frames:[first*10,last*10],kind:'flagged',reasons:['low_iou'],reviewed:false});
        window.fixture = {revision:'fixture',segments:[segment(0,6),segment(8,15),segment(17,25)],summary:{unprocessed_frames:0,flagged_frames:24,reviewed_frames:0}};
        window.original = JSON.stringify(fixture);
        window.api = async () => structuredClone(fixture);
        window.post = async (_, bounds) => {
          fixture.segments.find(s => s.first === bounds.first && s.last === bounds.last).reviewed = true;
          return structuredClone(fixture);
        };
        window.serverIsApp = window.isWorkspace = () => true;
        window.currentSourceKey = () => 'fixture';
        window.frameOfRow = row => row * 10;
        window.maskEditor = {requestLeave:async () => true};
        window.setRegionRows = (first,last) => {state.region={first,last};};
        window.showRow = async row => {state.row=row;};
        window.syncTaskSelection = window.showTab = () => {};
        window.readStorage = (key,fallback) => JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback));
        window.writeStorage = (key,value) => localStorage.setItem(key,JSON.stringify(value));
      });
      await page.addScriptTag({path:path.resolve('src/worm_pose_gen/pose_viewer_ui/workflow_inspect.js')});
      await page.evaluate(async () => {workflowInspect.init(); await workflowInspect.refresh();});
    };
    await mount();
    const input = page.locator('#inspection-min-frames'), queue = page.locator('.inspection-segment');
    assert.equal(await input.inputValue(), '8');
    assert.equal(await queue.count(), 2);
    assert.match(await queue.first().textContent(), /Frames 80–150 · 8 frames/);
    await page.locator('#inspection-next').click();
    assert.equal(await page.evaluate(() => state.region.first), 8);
    await page.locator('#inspection-mark').click();
    await page.waitForFunction(() => state.region.first === 17);
    await page.locator('#inspection-mark').click();
    await page.waitForFunction(() => document.querySelector('#inspection-status').textContent.includes('Lower the minimum'));
    assert.equal(await page.evaluate(() => fixture.segments[0].reviewed), false, '7 sampled frames remain unreviewed despite spanning 61 source frames');
    assert(await page.locator('#inspection-next').isDisabled());
    await page.locator('#inspection-show-shorter').click();
    assert.equal(await input.inputValue(), '1');
    assert.equal(await queue.count(), 1);
    await page.locator('#inspection-next').click();
    assert.equal(await page.evaluate(() => state.region.first), 0);
    await input.fill('9');
    await page.locator('#inspection-show-reviewed').check();
    assert.equal(await queue.count(), 1, 'inclusive threshold hides the eight-frame segment');
    await input.fill('0');
    assert.equal(await input.getAttribute('aria-invalid'), 'true');
    assert.equal(await queue.count(), 1, 'invalid input preserves last valid filter');
    await input.fill('8');
    assert.equal(await input.getAttribute('aria-invalid'), null);
    await input.fill('10');
    assert.equal(await queue.count(), 0);
    assert.match(await page.locator('#inspection-segments').textContent(), /minimum of 10 frames/);
    await mount();
    assert.equal(await input.inputValue(), '10', 'minimum survives page reload');
    assert.equal(await queue.count(), 0);
    assert.equal(await page.evaluate(() => JSON.stringify(fixture) === original), true, 'filter does not mutate segments, flags, or reviews');
    assert.deepEqual(errors, []);
    console.log('PASS Inspect default8, sampled-frame thresholds, skip/reviewnext filtering, shorter recovery, validation and persistence');
  } finally { await browser.close(); }
})();
