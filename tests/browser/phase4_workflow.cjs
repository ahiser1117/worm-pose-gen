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
    assert.deepEqual(await page.locator('#tabs button').allTextContents(), ['Run', 'Inspect', 'Paint', 'Compare', 'Export']);
    // Global navigation stays at the left; file actions have separate centered panels.
    assert.deepEqual(await page.locator('#app-header .file-actions button, #screen-tabs button, #toggle-inspector').allTextContents(), ['Import', 'Open', 'Workspace', 'Labels', 'Training', 'Side panel']);
    for (const tab of ['labels', 'training', 'workspace']) {
      await page.locator(`[data-main-tab="${tab}"]`).click();
      assert.equal(await page.locator('#screen-tabs [aria-selected="true"]').textContent(), tab[0].toUpperCase() + tab.slice(1));
      assert.equal(await page.evaluate(() => state.screen), tab);
    }
    await page.locator('#app-header [data-screen="open"]').click();
    assert(await page.locator('#screen-open').isVisible());
    assert(!(await page.locator('#screen-import').isVisible()));
    const panel = await page.locator('#screen-open').boundingBox();
    assert(Math.abs(panel.x + panel.width / 2 - page.viewportSize().width / 2) < 2, 'Open panel is horizontally centered');
    await page.locator('#app-header [data-screen="import"]').click();
    await page.waitForFunction(root => document.querySelector('#explorer-path').value === root, root);
    assert(!(await page.locator('#screen-open').isVisible()));
    assert(await page.locator('#explorer').isVisible());
    await page.locator('#explorer-up').click();
    await page.waitForFunction(() => document.querySelector('#explorer-path').value === '/tmp');
    await page.locator('#explorer-path').fill(root + '/runs');
    await page.locator('#explorer-path').press('Enter');
    await page.waitForFunction(root => document.querySelector('#explorer-up').dataset.parent === root, root);
    await page.locator('#explorer-entries button').filter({hasText: 'baseline'}).click();
    await page.waitForFunction(root => document.querySelector('#explorer-path').value === root + '/runs/baseline', root);
    await page.locator('#explorer-breadcrumbs button').filter({hasText: path.basename(root)}).click();
    await page.waitForFunction(root => document.querySelector('#explorer-path').value === root, root);
    await page.locator('#explorer-breadcrumbs button').filter({hasText: /^\/$/}).click();
    await page.waitForFunction(() => document.querySelector('#explorer-path').value === '/');
    assert(await page.locator('#explorer-up').isDisabled());
    await page.locator('#explorer-path').fill(root);
    await page.locator('#explorer-go').click();
    await page.waitForFunction(root => document.querySelector('#explorer-path').value === root && document.querySelector('#explorer-up').dataset.parent === '/tmp', root);
    // Import a video with no calibration cache and keep preparation pending long
    // enough to assert the loading UI is visible before the preview can appear.
    fs.copyFileSync(path.join(root, 'recording.h5'), path.join(root, 'fresh.h5'));
    await page.locator('#explorer-go').click();
    await page.locator('#explorer-entries button').filter({hasText: 'fresh.h5'}).click();
    await page.waitForFunction(() => !document.querySelector('#explorer-register').disabled && document.querySelector('#explorer-file-name').textContent.includes('fresh.h5'));
    let releasePreparation;
    const preparationGate = new Promise(resolve => { releasePreparation = resolve; });
    await page.route('**/api/recordings/prepare', async route => { await preparationGate; await route.continue(); }, {times: 1});
    await page.locator('#explorer-register').click();
    await page.waitForFunction(() => document.querySelector('#preview-progress-steps').textContent.includes('Preparing illumination correction'));
    assert(await page.locator('#preview-progress').isVisible());
    assert(await page.locator('#explorer-register').isDisabled());
    assert(!fs.existsSync(path.join(root, 'cache/flat_fields/fresh.npz')));
    await page.screenshot({path: '/tmp/worm-import-loading.png'});
    releasePreparation();
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('fresh imported'));
    assert(fs.existsSync(path.join(root, 'cache/flat_fields/fresh.npz')));
    assert(!(await page.locator('#preview-progress').isVisible()));
    assert(await page.locator('#recording-thumb').isVisible());
    await page.screenshot({path: '/tmp/worm-import-panel.png'});
    await page.locator('#screen-import [data-close-panel]').click();
    assert.equal(await page.evaluate(() => state.screen), 'workspace');
    await page.setViewportSize({width: 480, height: 800});
    await page.locator('#app-header [data-screen="import"]').click();
    assert(await page.locator('#explorer-go').isVisible());
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.locator('#screen-import [data-close-panel]').click();
    await page.setViewportSize({width: 1500, height: 1050});
    // Human review is explicit and leaves automatic flags intact.
    await page.locator('#tabs [data-tab="inspect"]').click();
    await page.locator('#inspection-min-frames').fill('1');
    await page.waitForFunction(() => document.querySelectorAll('#inspection-segments .inspection-segment').length > 0);
    const beforeReview = await get('/api/workspaces/demo/inspection');
    await page.locator('#inspection-segments .inspection-segment').first().click();
    const reviewResponse = responseTo('/inspection/review');
    await page.locator('#inspection-mark').click();
    assert.equal((await reviewResponse).status(), 200);
    const afterReview = await get('/api/workspaces/demo/inspection');
    assert(afterReview.summary.reviewed_frames > 0);
    assert.equal(afterReview.summary.flagged_frames, beforeReview.summary.flagged_frames);
    const paint = async (frame, split) => {
      if (await page.evaluate(() => state.screen !== "workspace")) await page.locator("#return-workspace").click();
      await page.locator('#tabs [data-tab="paint"]').click();
      await page.locator('#frame-index').fill(String(frame)); await page.locator('#go').click();
      await page.waitForFunction(f => document.querySelector('#mask-status').textContent.startsWith(`Frame ${f}:`), frame);
      assert.equal(await page.locator('#tab-paint details').count(), 0);
      for (const id of ['mask-threshold','mask-combine','mask-next-mode','mask-stride','mask-candidates','mask-save-next']) assert(await page.locator('#'+id).isVisible(), id);
      await page.locator('#mask-enable').check(); await page.locator('[data-mask-brush="127"]').click();
      const point = await page.evaluate(() => { const r = canvas.getBoundingClientRect(), v = state.view; return {x:r.x+v.tx+30*v.scale, y:r.y+v.ty+30*v.scale}; });
      await page.mouse.click(point.x, point.y);
      assert(await page.evaluate(() => maskEditor.isDirty()));
      const draftView = await page.evaluate(() => ({...state.view}));
      await page.locator('#tabs [data-tab="inspect"]').click();
      await page.locator('#tabs [data-tab="paint"]').click();
      assert(await page.evaluate(() => maskEditor.isDirty()));
      assert.deepEqual(await page.evaluate(() => ({...state.view})), draftView);
      const saved = responseTo('/mask'); await page.locator('#mask-save').click(); assert.equal((await saved).status(), 200);
      await page.waitForFunction(() => !document.querySelector('#mask-to-corpus').disabled);
      if (split) {
        await page.locator('#mask-also-corpus').check();
        await page.locator('#mask-corpus-split').selectOption(split);
        const savedLabel = responseTo('/labeling/save'); await page.locator('#mask-to-corpus').click(); assert.equal((await savedLabel).status(), 200);
        await page.waitForFunction(() => !document.querySelector('#mask-also-corpus').disabled);
        await page.locator('#mask-also-corpus').uncheck();
      }
    };
    await paint(0, 'train');
    assert.equal((await get('/api/workspaces/demo/mask?frame=0')).stale, true);
    assert.deepEqual((await get('/api/workspaces/demo/inspection')).reviewed_rows, afterReview.reviewed_rows, 'Painting a distant frame preserves reviewed segments');
    await page.locator('#app-header [data-screen="labels"]').click();
    await page.locator('#corpus-list button').filter({hasText:'Open'}).first().click();
    await page.waitForFunction(() => maskEditor.corpusActive());
    await page.locator('[data-mask-brush="255"]').click();
    const point = await page.evaluate(() => { const r=canvas.getBoundingClientRect(),v=state.view; return {x:r.x+v.tx+45*v.scale,y:r.y+v.ty+30*v.scale}; });
    await page.mouse.click(point.x, point.y);
    await page.locator('#mask-save').click();
    await page.waitForFunction(() => !maskEditor.isDirty() && document.querySelector('#mask-save-status').textContent.includes('saved'));
    assert.equal((await get('/api/corpus')).samples[0].revision, 2);
    await page.locator('#mask-corpus-close').click();
    await paint(1, 'val');
    await page.locator('#app-header [data-screen="training"]').click();
    await page.locator('#tune-epochs').fill('1'); await page.locator('#tune-batch').fill('1');
    await page.locator('.training-advanced summary').click();
    await page.locator('#tune-crop').fill('64'); await page.locator('#tune-device').selectOption('cpu');
    await page.locator('#corpus-train').click(); await waitJob('fine_tune');
    await page.locator('#app-header [data-screen="training"]').click(); await page.locator('#checkpoint-refresh').click();
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
    await page.locator('#tabs [data-tab="review"]').click();
    await page.locator('.cand-set [data-show]').first().click();
    await page.waitForFunction(() => shownDetail(0) !== null);
    await page.locator('#compare-side-toggle').check();
    assert(await page.locator('#workspace-main > #compare-side').isVisible());
    await page.screenshot({path:'/tmp/worm-workflow-compare.png'});
    await page.locator('#workflow-compare [data-overlay="A"]').click();
    assert.equal(await page.evaluate(() => workflowCompare.visible('Current')), false);
    await page.locator('#tabs [data-tab="inspect"]').click();
    assert(!(await page.locator('#compare-side').isVisible()));
    assert.equal(await page.evaluate(() => workflowCompare.visible('Current')), true);
    await page.locator('#tabs [data-tab="review"]').click();
    await page.locator('#workflow-compare [data-overlay="all"]').click();
    const accepted = responseTo('/accept'); await page.locator('[data-accept-slot="0"]').click(); assert.equal((await accepted).status(),200);
    const fitted = await get('/api/workspaces/demo/frame?frame=2'); assert.equal(fitted.stats.fitted,true); assert.equal(fitted.mask_stale,false);
    await page.locator('#tabs [data-tab="paint"]').click();
    await page.locator('#right-tab-history').click();
    const undone = responseTo('/edits'); await page.locator('#edit-undo').click(); assert.equal((await undone).status(),200);
    const restored = await get('/api/workspaces/demo/frame?frame=2'); assert.equal(restored.stats.fitted,false); assert.equal(restored.mask_stale,true);
    await page.locator('#tabs [data-tab="export"]').click();
    await page.locator('[data-export-name]').fill('reviewed-poses-v1');
    const exportedResponse = responseTo('/export');
    await page.locator('[data-export]').click();
    const exported = await exportedResponse;
    assert.equal(exported.status(), 200, await exported.text());
    const exportInfo = await exported.json();
    const download = await page.request.get(base + exportInfo.download_url);
    assert.equal(download.status(), 200);
    const parquet = await download.body();
    assert.equal(parquet.subarray(0, 4).toString(), 'PAR1');
    assert.equal(parquet.subarray(-4).toString(), 'PAR1');
    assert(fs.existsSync(path.join(exportInfo.snapshot_path, 'export.json')));
    assert.equal(exportInfo.rows, 6);
    await page.screenshot({path:"/tmp/worm-workflow-export.png"});
    assert.deepEqual(errors, []);
    console.log('Phase 4 browser workflow passed: paint, corpus revisions, training, checkpoint selection, region job, accept, undo, human review and named snapshot export.');
  } catch (error) { console.error(logs); throw error; }
  finally {
    if (browser) await browser.close();
    if (server.exitCode === null) { server.kill('SIGTERM'); await new Promise(resolve => server.once('exit', resolve)); }
    fs.rmSync(root, {recursive:true, force:true});
  }
})().catch(error => { console.error(error); process.exitCode=1; });
