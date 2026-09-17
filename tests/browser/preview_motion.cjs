// Run against phase4_fixture.py. Delays affect only this browser's preview APIs.
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.UI_BASE_URL || 'http://127.0.0.1:18779');
    await page.waitForFunction(() => state.runName === 'demo' && state.frame?.detail === 'full');
    const template = await page.evaluate(() => {
      abortLoad('prefetch');
      const fields = Object.fromEntries(Object.entries(state.frame).filter(([key]) => !key.startsWith('_')));
      return structuredClone(fields);
    });
    template.candidate_sets = [{id: 'motion-option', algorithm: 'mirror', index: 0, stale: false}];
    template.pose.candidate_sets = template.candidate_sets;
    let fullDelay = 400, lightDelay = 20, compareDelay = 80;
    await page.route('**/api/workspaces/motion-reference/pose?**', async route => {
      const row = Number(new URL(route.request().url()).searchParams.get('frame'));
      requests.push({detail: 'compare', row});
      await sleep(compareDelay);
      await route.fulfill({json: {present: true, pose: template.pose, stats: {...template.stats, row}}}).catch(() => {});
    });
    await page.route('**/api/workspaces/demo/frame?**', async route => {
      const url = new URL(route.request().url()), detail = url.searchParams.get('detail'), row = Number(url.searchParams.get('frame'));
      requests.push({detail, row});
      const payload = structuredClone(template);
      Object.assign(payload, {row, frame_index: row, detail, details_deferred: detail === 'light'});
      payload.stats.row = row;
      payload.stats.frame_index = row;
      if (detail === 'light') {
        payload.layers = {image: payload.layers.image};
        payload.mask_stats = null;
        for (const key of ['has_stored_mask', 'has_override', 'mask_revision', 'candidate_sets']) delete payload[key];
        delete payload.pose.candidate_sets;
      }
      await sleep(detail === 'full' ? fullDelay : lightDelay);
      await route.fulfill({json: payload}).catch(() => {}); // superseded fetches are aborted
    });
    await page.evaluate(() => {
      window.pixelComposites = 0; window.pixelDecodes = 0;
      const create = overlayCtx.createImageData.bind(overlayCtx);
      overlayCtx.createImageData = (...args) => { pixelComposites++; return create(...args); };
      const decode = decodeGray;
      decodeGray = (...args) => { pixelDecodes++; return decode(...args); };
      window.resetMotionCounters = () => { pixelComposites = 0; pixelDecodes = 0; };
      window.clearPreviewCache = () => {
        abortLoad('prefetch'); abortLoad('light'); abortLoad('full'); cancelPendingFull();
        loads.lightWanted = null; loads.generation++; state.frameCache.clear();
      };
      showTab('inspect');
      document.querySelector('#fps').value = '2';
      document.querySelector('#loop-stretch').checked = true;
      state.region = {first: 0, last: 4};
      state.compareName = 'motion-reference'; state.compareKind = 'workspace';
      state.candidateSets = [{id: 'motion-option', algorithm: 'mirror', rows: [0, 5], stale: false}];
      clearPreviewCache();
    });
    // Establish a cached full frame, then verify playback suppresses its layers.
    await page.evaluate(() => showRow(0, {keepView: true, immediate: true}));
    assert(await page.evaluate(() => !!state.comparePose));
    assert(await page.locator('#frame-candidates-section').isVisible());
    requests.length = 0;
    await page.evaluate(() => { resetMotionCounters(); togglePlay(); });
    await page.waitForFunction(() => state.frame?.row >= 1 && state.frame.detail === 'light');
    await sleep(650); // slower than both the ordinary debounce and delayed full response
    assert(requests.every(r => r.detail === 'light'), 'Playback must never request full layers or comparison validation');
    assert.equal(await page.evaluate(() => state.comparePose), null);
    assert(await page.locator('#frame-candidates-section').isHidden());
    assert.deepEqual(await page.evaluate(() => [pixelComposites, pixelDecodes]), [0, 0]);
    assert(await page.evaluate(() => layer('tube').on), 'Deferring must preserve the selected layers');
    assert(!/Tube outline/.test(await page.locator('#legend').innerText()));
    assert.match(await page.locator('#caption').innerText(), /quick preview/);
    await page.locator('#play').click();
    const pausedRow = await page.evaluate(() => state.row);
    await page.waitForFunction(() => !state.playing && state.frame?.row === state.row && state.frame.detail === 'full');
    await sleep(200);
    assert.deepEqual(requests.filter(r => r.detail === 'full').map(r => r.row), [pausedRow]);
    assert.deepEqual(requests.filter(r => r.detail === 'compare').map(r => r.row), [pausedRow]);
    assert(await page.evaluate(() => !!state.comparePose));
    assert(await page.locator('#frame-candidates-section').isVisible());
    assert(await page.evaluate(() => pixelComposites > 0 && pixelDecodes > 0));
    assert.match(await page.locator('#legend').innerText(), /Tube outline/);

    // A held scrubber must stay light even after the pointer stops moving.
    await page.evaluate(() => { clearPreviewCache(); resetMotionCounters(); });
    requests.length = 0;
    const chart = page.locator('#charts canvas').first();
    await chart.scrollIntoViewIfNeeded();
    const box = await chart.boundingBox();
    const atRow = row => ({x: box.x + box.width * (row + .5) / 6, y: box.y + box.height / 2});
    const first = atRow(1), last = atRow(4);
    await page.mouse.move(first.x, first.y); await page.mouse.down();
    await page.mouse.move(last.x, last.y, {steps: 4});
    await page.waitForFunction(() => state.timeline.dragging && state.frame?.row === 4);
    await sleep(600);
    assert(requests.every(r => r.detail === 'light'), 'Holding a scrub gesture must not trigger detailed rendering');
    assert.equal(await page.evaluate(() => state.comparePose), null);
    assert(await page.locator('#frame-candidates-section').isHidden());
    assert.deepEqual(await page.evaluate(() => [pixelComposites, pixelDecodes]), [0, 0]);
    await page.mouse.up();
    await page.waitForFunction(() => !state.timeline.dragging && state.frame?.row === 4 && state.frame.detail === 'full');
    assert.deepEqual(requests.filter(r => r.detail === 'full').map(r => r.row), [4]);

    // Cached full frames must also be projected to image + vectors when scrubbing.
    requests.length = 0;
    await page.evaluate(() => resetMotionCounters());
    await page.mouse.move(last.x, last.y); await page.mouse.down();
    await page.mouse.move(first.x, first.y); await page.mouse.move(last.x, last.y);
    await page.waitForFunction(() => state.frame?.row === 4 && state.frame.detail === 'light');
    await sleep(250);
    assert(await page.evaluate(() => rasterLayersDeferred()));
    assert(await page.evaluate(() => state.frame.details_deferred && !('candidate_sets' in state.frame) && !('candidate_sets' in state.frame.pose)));
    assert(await page.locator('#frame-candidates-section').isHidden());
    assert.deepEqual(await page.evaluate(() => [pixelComposites, pixelDecodes]), [0, 0]);
    await page.mouse.up();
    await page.waitForFunction(() => !rasterLayersDeferred());
    assert.equal(requests.filter(r => r.detail === 'full').length, 0, 'Paused cache hit needs no request');

    // A late full response must not paint during motion or replace the chosen row.
    compareDelay = 400;
    await page.evaluate(() => { clearPreviewCache(); showRow(1, {keepView: true, immediate: true}); });
    await page.waitForFunction(() => !!loads.full);
    await page.evaluate(() => { window.pendingCompare = loads.compare; });
    await page.mouse.move(last.x, last.y); await page.mouse.down();
    assert(await page.evaluate(() => pendingCompare.signal.aborted));
    await page.waitForFunction(() => state.frame?.row === 4 && state.frame.detail === 'light');
    await sleep(500);
    assert.equal(await page.evaluate(() => state.frame.row), 4);
    assert.equal(await page.evaluate(() => state.frame.detail), 'light');
    assert.equal(await page.evaluate(() => state.comparePose), null, 'An old comparison response cannot return during motion');
    await page.mouse.up();
    await page.waitForFunction(() => state.frame?.row === 4 && state.frame.detail === 'full');

    // Ending playback on the last frame and losing pointer capture both settle.
    await page.evaluate(() => { clearPreviewCache(); state.row = 4; state.region = null; document.querySelector('#loop-stretch').checked = false; document.querySelector('#fps').value = '10'; togglePlay(); });
    await page.waitForFunction(() => state.row === 5 && !state.playing && state.frame?.row === 5 && state.frame.detail === 'full');
    await page.evaluate(() => clearPreviewCache());
    await page.mouse.move(first.x, first.y); await page.mouse.down();
    await page.waitForFunction(() => state.timeline.dragging && state.row === 1);
    await chart.dispatchEvent('pointercancel');
    await page.mouse.up();
    await page.waitForFunction(() => !state.timeline.dragging && state.frame?.row === 1 && state.frame.detail === 'full');

    // Older light responses cannot roll back a selected frame or downgrade it.
    lightDelay = 650; fullDelay = 40;
    await page.evaluate(() => { clearPreviewCache(); showRow(2, {keepView: true}); });
    await page.waitForFunction(() => loads.lightRow === 2 && !!loads.light);
    await page.evaluate(() => showRow(3, {keepView: true, immediate: true}));
    await page.waitForFunction(() => state.frame?.row === 3 && state.frame.detail === 'full');
    await sleep(1400);
    assert.equal(await page.evaluate(() => state.frame.row), 3);
    assert.equal(await page.evaluate(() => state.frame.detail), 'full');
    assert.deepEqual(errors, []);
    console.log('Preview motion: cheap playback/scrubbing, deferred candidates/comparison, full and light cache hits, deferred pixel decoding/compositing, pause/release/cancel/end restoration, stale response protection passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
