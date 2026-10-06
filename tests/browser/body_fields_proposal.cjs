// Run from the repo root with Playwright + Chromium available (same env as mask_editor.cjs).
// Starts tests/browser/body_fields_fixture.py --proposals (a synthetic CPU app whose two records carry a
// network proposal) and drives proposals in Body fields: the proposal layer and summary, accept with G
// (advancing), Edit proposal into trace mode with draggable, insertable and removable points, and Propose
// without a configured network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-body-proposals-'));
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/body_fields_fixture.py', '--root', root, '--port', String(port), '--proposals'], {
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
    const record = async id => (await fetch(`${base}/api/body-fields/${id}`)).json();
    await page.goto(base);
    await page.locator('[data-main-tab="bodyfields"]').click();
    await page.waitForFunction(() => document.querySelectorAll('#bf-list .bf-row').length === 3 && bodyFields.current()?.proposal);
    assert.match(await text('#bf-summary'), /Proposal: IoU 0\.910 vs current 0\.970 · 5 points · stub\.ckpt/);
    assert.equal(await page.locator('#bf-list .bf-badge.proposal').count(), 2);
    assert.match(await text('#bf-list .bf-row:first-child'), /proposal IoU 0\.910/);
    await page.locator('#bf-proposal').selectOption('no');
    await page.waitForFunction(() => document.querySelectorAll('#bf-list .bf-row').length === 1);
    await page.locator('#bf-proposal').selectOption('');
    await page.waitForFunction(() => document.querySelectorAll('#bf-list .bf-row').length === 3);

    // Image pixel <-> screen, for the 96x64 fixture fitted into the canvas.
    const screen = async (x, y) => {
      const box = await page.locator('#bf-canvas').boundingBox();
      const scale = Math.min(box.width / 96, box.height / 64);
      return {x: box.x + (box.width - 96 * scale) / 2 + (x + 0.5) * scale, y: box.y + (box.height - 64 * scale) / 2 + (y + 0.5) * scale};
    };
    const hover = async (x, y) => { const p = await screen(x, y); await page.mouse.move(p.x, p.y); return text('#bf-readout'); };
    // The record runs head at x=10; the proposal the other way. Its A-P layer replaces the record's on demand.
    assert.match(await hover(85, 32), /· A-P 0\.9/);
    await page.locator('[data-bf-layer="proposal_ap"] input[type=checkbox]').check();
    assert.match(await hover(85, 32), /proposal A-P 0\.0/);
    await page.locator('[data-bf-layer="proposal_ap"] input[type=checkbox]').uncheck();

    // G accepts the proposal without a refit and advances to the next sample.
    await page.locator('#bf-canvas').click({position: {x: 5, y: 5}});
    await page.keyboard.press('g');
    await page.waitForFunction(() => document.querySelector('#bf-caption').textContent.startsWith('rec_f000002') && bodyFields.current()?.proposal);
    const accepted = await record('rec_f000001');
    assert.deepEqual([accepted.meta.fit_method, accepted.meta.trace_source, accepted.meta.review, accepted.meta.fit_iou], ['traced', 'network', 'accepted', 0.91]);
    assert.equal(accepted.proposal, undefined);
    assert.equal(Math.round(accepted.head_xy[0]), 90);
    assert.match(await text('#bf-list .bf-row:first-child'), /traced/);

    // Edit proposal: its trace points become editable; drag one, insert one on the line, remove one.
    await page.locator('#bf-proposal-edit').click();
    assert(await page.locator('#bf-trace-panel').isVisible());
    const points = () => page.evaluate(() => bodyFields.trace().points.map(p => p.map(Math.floor)));
    assert.deepEqual((await points()).map(p => p[0]), [90, 70, 50, 31, 11]);
    const from = await screen(50.3, 32), to = await screen(51, 26);
    await page.mouse.move(from.x, from.y); await page.mouse.down(); await page.mouse.move(to.x, to.y, {steps: 5}); await page.mouse.up();
    assert.deepEqual((await points())[2], [51, 26], 'a dragged point moves');
    const onLine = await screen(80, 32); await page.mouse.click(onLine.x, onLine.y);
    assert.deepEqual((await points()).map(p => p[0]), [90, 80, 70, 51, 31, 11], 'a click on the line inserts a point there');
    const tail = await screen(11.1, 32); await page.mouse.click(tail.x, tail.y, {button: 'right'});
    assert.deepEqual((await points()).map(p => p[0]), [90, 80, 70, 51, 31], 'right-click removes a point');
    const empty = await screen(40, 5), panTo = await screen(60, 15);
    await page.mouse.move(empty.x, empty.y); await page.mouse.down(); await page.mouse.move(panTo.x, panTo.y, {steps: 5}); await page.mouse.up();
    assert.equal((await points()).length, 5, 'a drag on empty canvas pans and adds nothing');
    await page.keyboard.press('0');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => /Fit along the trace: IoU new/.test(document.querySelector('#bf-trace-result').textContent), null, {timeout: 180000});
    await page.screenshot({path: path.join(os.tmpdir(), 'worm-body-fields-proposal-edit.png')});
    await page.locator('#bf-trace-accept').click();
    await page.waitForFunction(() => /Fit traced along 5 clicked points/.test(document.querySelector('#bf-summary').textContent), null, {timeout: 180000});
    const edited = await record('rec_f000002');
    assert.deepEqual([edited.meta.fit_method, edited.meta.review, edited.trace_xy.length], ['traced', 'accepted', 5]);

    // Propose needs the network; this app has none configured.
    await page.locator('#bf-list .bf-row:last-child').click();
    await page.waitForFunction(() => document.querySelector('#bf-caption').textContent.startsWith('rec_f000003'));
    assert(await page.locator('#bf-propose').isDisabled(), 'a sample without targets cannot be proposed');
    await page.locator('#bf-list .bf-row:first-child').click();
    await page.waitForFunction(() => document.querySelector('#bf-caption').textContent.startsWith('rec_f000001'));
    assert.equal(await text('#bf-propose'), 'Propose');
    await page.locator('#bf-propose').click();
    await page.waitForFunction(() => /Propose a trace: .*--body-net/.test(document.querySelector('#bf-edit-status').textContent));
    assert.deepEqual(errors, []);
    console.log('body field proposals: layer and summary, filter, accept with G and advance, edit proposal (drag, insert, remove, pan), preview and accept, propose without a network passed');
  } finally {
    if (browser) await browser.close();
    server.kill();
    fs.rmSync(root, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
