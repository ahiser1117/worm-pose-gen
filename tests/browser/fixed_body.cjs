// Real stage job, persisted overlay, playback payload and invalidation after a flip.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-fixed-body-'));
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn('.venv/bin/python', ['tests/browser/fixed_body_fixture.py', '--root', root, '--port', String(port)], {
    env: {...process.env, PYTHONPATH: `${process.cwd()}/src:${process.cwd()}`, CUDA_VISIBLE_DEVICES: '',
      OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1', MPLCONFIGDIR: path.join(root, 'mpl')},
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '', browser;
  server.stdout.on('data', b => { logs = (logs + b).slice(-10000); });
  server.stderr.on('data', b => { logs = (logs + b).slice(-10000); });
  try {
    let ready = false;
    for (let i = 0; i < 60; i++) {
      try { ready = (await fetch(base + '/api/state')).ok; } catch {}
      if (ready || server.exitCode !== null) break;
      await sleep(500);
    }
    assert(ready, logs);
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport: {width: 1500, height: 1050}}), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.waitForFunction(() => state.runName === 'body' && state.frame?.frame_index === 0);
    assert.equal(await page.evaluate(() => !!state.frame.fixed_body), false);
    await page.evaluate(() => {setRerunScope('workspace'); showTab('rerun');});
    await page.locator('#workflow-stage-details').evaluate(n => n.open = true);
    const stage = page.locator('#stages [data-stage="fixed_body"]');
    assert.equal(await stage.locator('.stage-include').isChecked(), false);
    const before = fs.readFileSync(path.join(root, 'workspaces/body/state.npz'));
    await stage.locator('.stage-run').click();
    await page.waitForFunction(() => state.frame?.fixed_body?.centerline_xy, null, {timeout: 60000});
    assert(before.equals(fs.readFileSync(path.join(root, 'workspaces/body/state.npz'))));
    await page.evaluate(async () => {showTab('inspect'); await showRow(4, {immediate:true}); openRightTab('layers');});
    await page.waitForFunction(() => state.frame?.frame_index === 4 && state.frame.fixed_body?.centerline_xy);
    const toggle = page.locator('.layer[data-layer="fixed_body"] input[type="checkbox"]');
    assert(await toggle.isChecked());
    const withBody = await page.locator('#canvas').screenshot();
    await toggle.uncheck();
    assert(!withBody.equals(await page.locator('#canvas').screenshot()), 'Fixed-body toggle changes the rendered canvas');
    await toggle.check();
    const light = await (await page.request.get(base + '/api/frame?run=body&frame=4&detail=light')).json();
    assert(light.fixed_body.extrapolated.some(Boolean));
    await page.evaluate(() => openRightTab('statistics'));
    assert.match(await page.locator('#stats').textContent(), /fixed length px/);
    await page.reload();
    await page.waitForFunction(() => state.frame?.fixed_body?.centerline_xy);
    await page.evaluate(async () => {await showRow(0, {immediate:true}); await flipFrame();});
    await page.waitForFunction(() => state.frame?.fixed_body?.stale);
    assert.equal(await page.evaluate(() => !!state.frame.fixed_body.centerline_xy), false);
    assert.match(await page.locator('#caption').textContent(), /fixed body outdated/);
    assert.deepEqual(errors, []);
    console.log('PASS optional fixed-body job, unchanged reviewed poses, visible toggle, extrapolation, persistence and edit invalidation');
  } catch (error) { console.error(logs); throw error; }
  finally { if (browser) await browser.close(); server.kill('SIGTERM'); await sleep(400); fs.rmSync(root, {recursive:true, force:true}); }
})();
