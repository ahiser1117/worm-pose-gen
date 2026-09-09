// Run from the repo root, with Playwright + Chromium available (same env as mask_editor.cjs).
// Launches its own synthetic CPU app and removes all test labels/checkpoints on completion.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-phase4-e2e-'));
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/phase4_fixture.py', '--root', root, '--port', String(port)], {
    env: {...process.env, PYTHONPATH: `${process.cwd()}/src:${process.cwd()}`, CUDA_VISIBLE_DEVICES: '',
      OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1', MPLCONFIGDIR: path.join(root, 'mpl')},
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '', browser;
  server.stdout.on('data', b => { logs = (logs + b).slice(-16000); });
  server.stderr.on('data', b => { logs = (logs + b).slice(-16000); });
  try {
    let ready = false;
    for (let i = 0; i < 60; i++) {
      try { ready = (await fetch(base + '/api/state')).ok; } catch {}
      if (ready || server.exitCode !== null) break;
      await sleep(500);
    }
    assert.ok(ready, logs);
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport: {width: 1500, height: 1050}}), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const get = async endpoint => { const response = await page.request.get(base + endpoint); assert.equal(response.status(), 200); return response.json(); };
    const responseTo = endpoint => page.waitForResponse(r => r.url().endsWith(endpoint) && r.request().method() === 'POST');
    const waitJob = async kind => {
      let job;
      for (let i = 0; i < 90; i++) {
        job = (await get('/api/jobs')).find(j => j.spec.kind === kind);
        if (job && ['done', 'failed', 'cancelled'].includes(job.state)) break;
        await sleep(500);
      }
      assert.equal(job?.state, 'done', JSON.stringify(job));
      return job;
    };
    await page.goto(base);
    await page.waitForFunction(() => state.runName === 'demo' && document.querySelector('#mask-status').textContent.startsWith('Frame 0:'));
    const paint = async (frame, split) => {
      await page.locator('[data-tab="view"]').click();
      await page.locator('#frame-index').fill(String(frame)); await page.locator('#go').click();
      await page.waitForFunction(f => document.querySelector('#mask-status').textContent.startsWith(`Frame ${f}:`), frame);
      await page.locator('#mask-enable').check(); await page.locator('[data-mask-brush="127"]').click();
      const point = await page.evaluate(() => { const r = canvas.getBoundingClientRect(), v = state.view; return {x:r.x+v.tx+30*v.scale, y:r.y+v.ty+30*v.scale}; });
      await page.mouse.click(point.x, point.y);
      const saved = responseTo('/mask'); await page.locator('#mask-save').click(); assert.equal((await saved).status(), 200);
      await page.waitForFunction(() => !document.querySelector('#mask-to-corpus').disabled);
      if (split) {
        await page.locator('#mask-corpus-split').selectOption(split);
        const savedLabel = responseTo('/corpus/labels'); await page.locator('#mask-to-corpus').click(); assert.equal((await savedLabel).status(), 200);
      }
    };
    await paint(0, 'train');
    assert.equal((await get('/api/workspaces/demo/mask?frame=0')).stale, true);
    await page.locator('[data-tab="corpus"]').click();
    await page.locator('#corpus-list button').filter({hasText:'Open'}).first().click();
    await page.waitForFunction(() => maskEditor.corpusActive());
    await page.locator('[data-mask-brush="255"]').click();
    const point = await page.evaluate(() => { const r=canvas.getBoundingClientRect(),v=state.view; return {x:r.x+v.tx+45*v.scale,y:r.y+v.ty+30*v.scale}; });
    await page.mouse.click(point.x, point.y);
    await page.locator('#mask-save').click();
    await page.waitForFunction(() => !maskEditor.isDirty() && document.querySelector('#mask-status').textContent.includes('revision 2'));
    assert.equal((await get('/api/corpus')).samples[0].revision, 2);
    await page.locator('#mask-corpus-close').click();
    await paint(1, 'val');
    await page.locator('[data-tab="corpus"]').click();
    await page.locator('#tune-epochs').fill('1'); await page.locator('#tune-batch').fill('1');
    await page.locator('#tune-crop').fill('64'); await page.locator('#tune-device').selectOption('cpu');
    await page.locator('#corpus-train').click(); await waitJob('fine_tune');
    await page.locator('[data-tab="corpus"]').click(); await page.locator('#checkpoint-refresh').click();
    await page.waitForFunction(() => [...document.querySelector('#seg-checkpoint').options].some(o=>o.value.endsWith(':best')));
    const checkpoint = await page.locator('#seg-checkpoint option').evaluateAll(opts => opts.find(o=>o.value.endsWith(':best')).value);
    await page.locator('#seg-checkpoint').selectOption(checkpoint); await page.locator('#checkpoint-select').click();
    await page.waitForFunction(() => document.querySelector('#checkpoint-status').textContent.startsWith('Selected'));
    const selected = (await get('/api/workspaces/demo')).selected_checkpoint;
    assert.match(selected, /runs\/.*\/best.ckpt$/);
    assert.equal(await page.evaluate(() => state.stageValues.segment.checkpoint), selected);
    const preview = await get('/api/workspaces/demo/frame?frame=1');
    assert.equal(preview.checkpoint, selected); assert.equal(preview.has_override, true);
    await paint(2);
    await page.locator('#mask-refit-frame').click(); await page.locator('#region-run').click(); await waitJob('region');
    await page.evaluate(() => loadCandidateSets());
    const accepted = responseTo('/accept'); await page.locator('.cand-set [data-accept]').first().click(); assert.equal((await accepted).status(),200);
    const fitted = await get('/api/workspaces/demo/frame?frame=2'); assert.equal(fitted.stats.fitted,true); assert.equal(fitted.mask_stale,false);
    await page.locator('[data-tab="view"]').click();
    const undone = responseTo('/edits'); await page.locator('#edit-undo').click(); assert.equal((await undone).status(),200);
    const restored = await get('/api/workspaces/demo/frame?frame=2'); assert.equal(restored.stats.fitted,false); assert.equal(restored.mask_stale,true);
    assert.deepEqual(errors, []);
    console.log('Phase 4 browser workflow passed: paint, corpus revisions, training, checkpoint selection, region job, accept and undo.');
  } catch (error) { console.error(logs); throw error; }
  finally {
    if (browser) await browser.close();
    if (server.exitCode === null) { server.kill('SIGTERM'); await new Promise(resolve => server.once('exit', resolve)); }
    fs.rmSync(root, {recursive:true, force:true});
  }
})().catch(error => { console.error(error); process.exitCode=1; });
