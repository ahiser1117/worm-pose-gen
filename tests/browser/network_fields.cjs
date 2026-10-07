// Run from the repo root with Playwright + Chromium available (same env as mask_editor.cjs).
// Starts tests/browser/network_fields_fixture.py (a fitted synthetic workspace and a stub body-field network)
// and checks the viewer's "Network A-P field" and "Network head / tail" layers: nothing is fetched while they
// are off, they draw and read out, a frame change predicts the new frame, a slow response for a frame no
// longer shown is dropped, and a failure shows a status.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-network-layers-'));
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/network_fields_fixture.py', '--root', root, '--port', String(port)], {
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
    const page = await browser.newPage({viewport: {width: 1400, height: 950}}), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const requested = [];
    page.on('request', request => { const m = request.url().match(/network-fields\?frame=(\d+)/); if (m) requested.push(Number(m[1])); });
    await page.goto(base);
    await page.waitForFunction(() => state.runName === 'demo' && state.frame?.detail === 'full');
    await page.evaluate(() => { showTab('inspect'); openRightTab('layers'); });
    const box = page.locator('[data-layer="net_ap"] input[type=checkbox]'), ends = page.locator('[data-layer="net_ends"] input[type=checkbox]');
    assert.equal(await box.isChecked(), false, 'off by default');
    assert.equal(await ends.isChecked(), false, 'off by default');
    await page.locator('#next').click(); await page.locator('#prev').click();
    await page.waitForFunction(() => state.row === 0 && state.frame?.detail === 'full');
    assert.deepEqual(requested, [], 'nothing is fetched while both layers are off');

    // Image pixel -> screen pixel of the stage canvas.
    const screen = (x, y) => page.evaluate(([x, y]) => { const r = canvas.getBoundingClientRect(), v = state.view; return {x: r.left + v.tx + (x + 0.5) * v.scale, y: r.top + v.ty + (y + 0.5) * v.scale}; }, [x, y]);
    const pixel = (x, y) => page.evaluate(([x, y]) => { const r = canvas.getBoundingClientRect(), v = state.view, d = devicePixelRatio;
      return Array.from(ctx.getImageData(Math.round((v.tx + (x + 0.5) * v.scale) * d), Math.round((v.ty + (y + 0.5) * v.scale) * d), 1, 1).data.slice(0, 3)); }, [x, y]);
    await page.evaluate(() => { for (const l of LAYERS) if (!['image', 'net_ap', 'net_ends'].includes(l.id)) { l.on = false; } renderLayers(); buildOverlay(); draw(); });
    const plain = await pixel(60, 48);
    await box.check();
    await page.waitForFunction(() => window.networkFields.shown()?.frame === 0);
    assert.deepEqual(requested, [0]);
    assert.match(await page.locator('#network-fields-status').textContent(), /Network · frame 0 · head peak 1\.00 · tail peak 1\.00/);
    const coloured = await pixel(60, 48);
    assert.notDeepEqual(coloured, plain, 'the A-P field is drawn');
    const at = await screen(60, 48);
    await page.mouse.move(at.x, at.y);
    assert.match(await page.evaluate(() => canvas.title), /network A-P 0\.47/);
    assert.match(await page.locator('#legend').textContent(), /Network A-P field/);

    // Ends: a green H at the head, a red T at the tail, off when the layer is off.
    const ring = async xy => { const p = await page.evaluate(([x, y]) => { const r = canvas.getBoundingClientRect(), v = state.view, d = devicePixelRatio;
      return Array.from(ctx.getImageData(Math.round((v.tx + x * v.scale + 4) * d), Math.round((v.ty + y * v.scale) * d), 1, 1).data.slice(0, 3)); }, xy); return p; };
    await box.uncheck();
    const beforeEnds = await ring([30, 48]);
    await ends.check();
    await page.waitForFunction(() => window.networkFields.shown());
    const head = await ring([30, 48]), tail = await ring([100, 48]);
    assert(head[1] > 200 && head[0] < 120, `head disc is green: ${head}`);
    assert(tail[0] > 200 && tail[1] < 120, `tail disc is red: ${tail}`);
    await ends.uncheck();
    assert.deepEqual(await ring([30, 48]), beforeEnds, 'no ends drawn with the layer off');
    await box.check();

    // The next frame is predicted; a slow response for a frame no longer shown is dropped.
    await page.locator('#next').click();
    await page.waitForFunction(() => window.networkFields.shown()?.frame === 1);
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    await page.route('**/network-fields?frame=2', async route => { await gate; await route.continue(); });
    await page.evaluate(() => { window.seen = []; window.watch = setInterval(() => window.seen.push(window.networkFields.shown()?.frame), 20); });
    await page.locator('#next').click();
    await page.waitForFunction(() => state.row === 2);
    await page.locator('#next').click();
    await page.waitForFunction(() => state.row === 3);
    assert.match(await page.locator('#network-fields-status').textContent(), /frame 1 \(updating\)/, 'the last prediction stays until the next arrives');
    release();
    await page.waitForFunction(() => window.networkFields.shown()?.frame === 3);
    assert.equal(await page.evaluate(() => { clearInterval(window.watch); return window.seen.includes(2); }), false, 'the frame-2 response was dropped');
    assert.deepEqual(requested, [0, 1, 2, 3], 'one request at a time, then the latest frame');

    // A failing request shows a status; with no network configured the layers are marked unavailable.
    await page.route('**/network-fields?frame=4', route => route.fulfill({status: 400, contentType: 'application/json',
      body: JSON.stringify({error: 'no body-field network checkpoint at /x; pass --body-net'})}));
    await page.locator('#next').click();
    await page.waitForFunction(() => /Network layers: .*--body-net/.test(document.querySelector('#network-fields-status').textContent));
    assert(await page.locator('.layer[data-layer="net_ap"]').evaluate(n => n.classList.contains('unavailable')));
    await page.locator('#next').click();
    await page.waitForFunction(() => state.row === 5);
    assert.deepEqual(requested, [0, 1, 2, 3, 4], 'an unconfigured network is not asked again on every frame');
    await page.screenshot({path: path.join(os.tmpdir(), 'worm-network-layers.png')});
    assert.deepEqual(errors, []);
    console.log('network layers: off by default and not fetched, A-P draw and readout, ends, next frame, stale response dropped, failure status passed');
  } finally {
    if (browser) await browser.close();
    server.kill();
    fs.rmSync(root, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
