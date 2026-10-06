// Run from the repo root with Playwright + Chromium available (same env as mask_editor.cjs).
// Starts tests/browser/body_fields_fixture.py (a synthetic CPU app) and drives the Body fields screen:
// browse and filter, layers and readout, temporal context, flip, review, trace midline, the Paint hand-off and a rebuild job.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-body-fields-'));
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/body_fields_fixture.py', '--root', root, '--port', String(port)], {
    env: {...process.env, PYTHONPATH: `${process.cwd()}/src:${process.cwd()}`, CUDA_VISIBLE_DEVICES: '', MPLCONFIGDIR: path.join(root, 'mpl')},
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '', browser;
  server.stdout.on('data', b => { logs = (logs + b).slice(-16000); });
  server.stderr.on('data', b => { logs = (logs + b).slice(-16000); });
  try {
    let ready = false;
    for (let i = 0; i < 60 && !ready && server.exitCode === null; i++) {
      try { ready = (await fetch(base + '/api/state')).ok; } catch { await sleep(500); }
    }
    assert.ok(ready, logs);
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport: {width: 1500, height: 1000}}), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const text = selector => page.locator(selector).textContent();
    await page.goto(base);
    await page.locator('[data-main-tab="bodyfields"]').click();
    assert.equal(await page.evaluate(() => state.screen), 'bodyfields');
    await page.waitForFunction(() => document.querySelectorAll('#bf-list .bf-row').length === 3);
    await page.waitForFunction(() => document.querySelector('#bf-context-status').textContent.startsWith('Frame t'));
    assert.match(await text('#bf-summary'), /nose landmark on this frame[\s\S]*IoU 0\.970/);
    assert.match(await text('#bf-list .bf-row:last-child'), /missing/);

    // Hovering image pixel (x, y) reports the label and the A-P value under it (head at x=10, tail at x=90).
    const hover = async (x, y) => {
      const box = await page.locator('#bf-canvas').boundingBox();
      const scale = Math.min(box.width / 96, box.height / 64);
      await page.mouse.move(box.x + (box.width - 96 * scale) / 2 + (x + 0.5) * scale, box.y + (box.height - 64 * scale) / 2 + (y + 0.5) * scale);
      return text('#bf-readout');
    };
    const ap = async (x, y) => Number((await hover(x, y)).match(/A-P ([\d.]+)/)[1]);
    assert.match(await hover(50, 32), /x 50 y 32 · worm · A-P 0\.5/);
    assert(await ap(15, 32) < 0.1);
    const pixel = () => page.evaluate(() => { const c = document.querySelector('#bf-canvas'), r = c.getBoundingClientRect(), d = devicePixelRatio;
      return Array.from(c.getContext('2d').getImageData(Math.round(r.width / 2 * d), Math.round(r.height / 2 * d), 1, 1).data.slice(0, 3)); });
    const coloured = await pixel();
    await page.locator('[data-bf-layer="ap"] input[type=checkbox]').uncheck();
    assert.notDeepEqual(await pixel(), coloured);
    await page.locator('[data-bf-layer="ap"] input[type=checkbox]').check();

    // Filters narrow the list on the server.
    await page.locator('#bf-max-iou').fill('0.9'); await page.locator('#bf-max-iou').dispatchEvent('change');
    await page.waitForFunction(() => document.querySelectorAll('#bf-list .bf-row').length === 1);
    assert.match(await text('#bf-list .bf-row'), /IoU 0\.600/);
    await page.locator('#bf-max-iou').fill(''); await page.locator('#bf-max-iou').dispatchEvent('change');
    await page.waitForFunction(() => document.querySelectorAll('#bf-list .bf-row').length === 3);

    // Temporal context: scrubbing shows the rolled neighbours; t-2 is outside the recording; lags clamp to max_lag.
    await page.locator('#bf-canvas').click();
    await page.keyboard.press('ArrowLeft'); await page.keyboard.press('ArrowLeft');
    assert.match(await text('#bf-context-status'), /t-2 is outside the recording/);
    await page.keyboard.press('ArrowRight');
    assert.match(await text('#bf-context-status'), /Frame t-1/);
    await page.keyboard.press('d');
    assert.match(await text('#bf-context-status'), /lag 2 has an invalid end|Lag 2 has an invalid end/);
    await page.keyboard.press('ArrowLeft');
    assert.match(await text('#bf-context-status'), /frame\[t\+1\] − frame\[t−1\]/);
    assert.match(await hover(9, 32), /Δ 140/);
    assert.match(await hover(91, 32), /Δ -140/);
    await page.keyboard.press('d');

    // Flip, then accept (which advances to the next sample).
    await page.keyboard.press('h');
    await page.waitForFunction(() => document.querySelector('#bf-summary').textContent.includes('set by hand'));
    assert(await ap(15, 32) > 0.9);
    await page.keyboard.press('a');
    await page.waitForFunction(() => document.querySelector('#bf-caption').textContent.startsWith('rec_f000002'));
    assert.match(await text('#bf-list .bf-row:first-child'), /accepted/);
    const record = await (await fetch(base + '/api/body-fields/rec_f000001')).json();
    assert.equal(record.meta.orientation, 'manual');
    assert.equal(record.meta.review, 'accepted');

    // The missing sample can only be rebuilt; a rebuild is a job, and its end is reported.
    await page.locator('#bf-list .bf-row:last-child').click();
    await page.waitForFunction(() => document.querySelector('#bf-summary').textContent.includes('No body fields yet'));
    assert(await page.locator('#bf-flip').isDisabled());
    await page.locator('#bf-rebuild').click();
    await page.waitForFunction(() => document.querySelector('#bf-edit-status').textContent.includes('queued'));
    assert(await page.locator('#bf-rebuild').isDisabled());
    const job = (await (await fetch(base + '/api/jobs')).json()).find(j => j.spec.kind === 'body_fields');
    await fetch(`${base}/api/jobs/${job.id}/cancel`, {method: 'POST'});
    await page.evaluate(() => loadJobs());
    await page.waitForFunction(() => /Rebuild (cancelled|failed)/.test(document.querySelector('#bf-edit-status').textContent));
    await page.waitForFunction(() => !document.querySelector('#bf-rebuild').disabled);

    // Trace midline: clicks add numbered points head first (a drag pans instead), Backspace removes
    // the last one, Enter previews the refit next to the record, and Accept writes it.
    await page.locator('#bf-list .bf-row').nth(1).click();
    await page.waitForFunction(() => document.querySelector('#bf-caption').textContent.startsWith('rec_f000002'));
    await page.locator('#bf-canvas').click({position: {x: 5, y: 5}});
    await page.keyboard.press('t');
    assert(await page.locator('#bf-trace-panel').isVisible());
    const point = async (x, y) => {
      const box = await page.locator('#bf-canvas').boundingBox(), scale = Math.min(box.width / 96, box.height / 64);
      return {x: box.x + (box.width - 96 * scale) / 2 + (x + 0.5) * scale, y: box.y + (box.height - 64 * scale) / 2 + (y + 0.5) * scale};
    };
    for (const x of [90, 70, 50, 30, 20]) { const p = await point(x, 32); await page.mouse.click(p.x, p.y); }
    await page.keyboard.press('Backspace');
    const last = await point(10, 32); await page.mouse.click(last.x, last.y);
    const from = await point(40, 10); await page.mouse.move(from.x, from.y); await page.mouse.down(); await page.mouse.move(from.x + 30, from.y + 10, {steps: 4}); await page.mouse.up();
    assert.deepEqual(await page.evaluate(() => bodyFields.trace().points.map(p => Math.floor(p[0]))), [90, 70, 50, 30, 10], 'a drag pans without adding a point');
    assert.match(await text('#bf-trace-count'), /5 points; point 1 is the head/);
    await page.keyboard.press('0');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => /IoU new \d\.\d{3} vs old 0\.600/.test(document.querySelector('#bf-trace-result').textContent), null, {timeout: 120000});
    const untouched = await (await fetch(base + '/api/body-fields/rec_f000002')).json();
    assert.equal(untouched.meta.fit_method, undefined, 'a preview writes nothing');
    assert(await ap(85, 32) < 0.2, 'the preview A-P field runs from the traced head');
    await page.screenshot({path: path.join(os.tmpdir(), 'worm-body-fields-trace.png')});
    await page.locator('#bf-trace-drawn').click();
    await page.waitForFunction(() => /Trace as drawn: IoU new/.test(document.querySelector('#bf-trace-result').textContent), null, {timeout: 120000});
    await page.locator('#bf-trace-accept').click();
    await page.waitForFunction(() => /Fit trace as drawn along 5 clicked points · automatic fit IoU was 0\.600/.test(document.querySelector('#bf-summary').textContent), null, {timeout: 120000});
    assert(await page.locator('#bf-trace-panel').isHidden());
    assert.match(await text('#bf-list .bf-row:nth-child(2)'), /trace as drawn/);
    const traced = await (await fetch(base + '/api/body-fields/rec_f000002')).json();
    assert.deepEqual([traced.meta.fit_method, traced.meta.review, traced.trace_xy.length], ['trace_as_drawn', 'accepted', 5]);
    await page.locator('#bf-method').selectOption('trace_as_drawn');
    await page.waitForFunction(() => document.querySelectorAll('#bf-list .bf-row').length === 1);
    await page.locator('#bf-method').selectOption('');
    await page.waitForFunction(() => document.querySelectorAll('#bf-list .bf-row').length === 3);
    await page.keyboard.press('t'); await page.keyboard.press('Escape');
    assert(await page.locator('#bf-trace-panel').isHidden());

    // Edit mask opens the corpus label in Paint; Body fields returns to the same sample.
    await page.locator('#bf-list .bf-row:first-child').click();
    await page.waitForFunction(() => document.querySelector('#bf-caption').textContent.startsWith('rec_f000001'));
    await page.locator('#bf-edit-mask').click();
    await page.waitForFunction(() => state.screen === 'paint' && paintScreen.current()?.target.sample_id === 'rec_f000001');
    assert.equal(await page.locator('#paint-back').textContent(), 'Return to Body fields');
    await page.locator('#paint-back').click();
    assert.equal(await page.evaluate(() => state.screen), 'bodyfields');
    await page.waitForFunction(() => document.querySelector('#bf-caption').textContent.startsWith('rec_f000001'));
    await page.screenshot({path: path.join(os.tmpdir(), 'worm-body-fields.png')});
    assert.deepEqual(errors, []);
    console.log('body fields: browse, filters, layers, readout, context scrub and differences, flip, review, rebuild job, trace midline, Paint hand-off passed');
  } finally {
    if (browser) await browser.close();
    server.kill();
    fs.rmSync(root, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
