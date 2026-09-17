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

    await page.locator('#app-header [data-screen="import"]').click();
    await page.waitForFunction(() => state.recordings.length > 0);
    await page.evaluate(() => selectRecording(state.recordings.find(r => r.name === 'recording').path));
    assert.equal(await page.locator('#ws-range-mode').inputValue(), 'entire');
    const total = await page.evaluate(() => state.recording.frames);
    assert.equal(await page.locator('#ws-last').inputValue(), String(total - 1));
    assert(await page.locator('#ws-first').isDisabled());
    await page.locator('#ws-range-mode').selectOption('selected');
    await page.locator('#ws-first').fill('1');
    await page.locator('#ws-last').fill('3');
    await page.locator('#ws-step').fill('2');
    assert((await page.locator('#ws-range-summary').textContent()).includes('2 frames'));
    await page.locator('#ws-name').fill('handoff');
    await page.locator('#ws-create').click();
    await page.waitForFunction(() => state.runName === 'handoff' && state.activeTask === 'rerun');
    assert.equal(await page.locator('#workflow-run-status').textContent(), 'This recording has not been processed yet');
    assert.equal(await page.evaluate(() => state.run.series.frame_index.join(',')), '1,3');
    assert.equal(await page.locator('#stages [data-stage="export"]').count(), 0);
    await page.evaluate(async () => { await selectSource('workspace', 'demo'); setRerunScope('workspace'); showTab('rerun'); });
    // An existing but unavailable custom checkpoint must block both run entry points.
    await page.evaluate(() => {
      state.stageValues.segment = {...state.stageValues.segment, checkpoint:'/tmp/nonexistent-workflow-test.ckpt'};
      rebuildStages(); workflowRun.render();
    });
    await page.waitForFunction(() => document.querySelector('#workflow-run-reason').textContent.includes('choose an available checkpoint'));
    assert(await page.locator('#run-all').isDisabled());
    assert(await page.locator('#stages [data-stage="segment"] .stage-run').isDisabled());
    assert(!/ready/i.test(await page.locator('#workflow-stage-progress').textContent()));
    assert(await page.locator('#workflow-model-select option').count() > 0);
    await page.locator('#workflow-model-use').click();
    await page.waitForFunction(() => document.querySelector('#workflow-run-reason').textContent.includes('Selected model is available'));
    assert.equal(await page.evaluate(() => document.querySelector('#rerun-scope').getBoundingClientRect().top < document.querySelector('#workflow-run-overview').getBoundingClientRect().top), true);
    await page.locator('#workflow-stage-details').evaluate(n => n.open = true);
    await page.locator('#stages .stage-include').evaluateAll(nodes => nodes.forEach(n => n.checked = n.closest('.stage').dataset.stage === 'track'));
    await page.evaluate(() => openRightTab('jobs'));
    await page.locator('#jobs-state').selectOption('running');
    await page.evaluate(() => togglePanel('right'));
    await page.locator('#run-all').click();
    await page.waitForFunction(() => state.rightTab === 'jobs' && !layout.collapsed.right);
    await page.waitForFunction(() => document.querySelector('#workflow-run-status').textContent === 'Pipeline complete', null, {timeout: 90000});
    assert(await page.locator('#workflow-inspect-results').isVisible());
    assert.equal(await page.evaluate(() => state.jobs.some(j => j.state === 'done' && jobStage(j) === 'track')), true);
    await page.locator('#workflow-inspect-results').click();
    assert.equal(await page.evaluate(() => state.activeTask), 'inspect');
    assert.equal(await page.locator('#inspection-min-frames').inputValue(), '8');
    await page.locator('#inspection-min-frames').fill('1');
    await page.waitForFunction(() => document.querySelectorAll('.inspection-segment').length > 0);
    const segment = page.locator('.inspection-segment').first();
    assert.match(await segment.textContent(), /\d+ frames?.*Needs human review/);
    assert.match(await segment.textContent(), /low iou|self contact|No fitted pose/);
    await segment.click();
    const reviewBounds = await page.evaluate(() => ({...state.region}));
    await page.locator('#inspection-mark').click();
    await page.waitForFunction(() => /All attention segments have been reviewed/.test(document.querySelector('#inspection-status').textContent) || document.querySelector('#inspection-target').textContent.includes('Selected frames'));
    await page.waitForFunction(count => document.querySelector('#inspection-summary').textContent.includes(`${count} human reviewed`), reviewBounds.last - reviewBounds.first + 1);
    await page.locator('#inspection-show-reviewed').check();
    assert.match(await page.locator('.inspection-segment').first().textContent(), /Human reviewed/);
    await page.evaluate(() => showTab('paint'));
    for (const height of [800, 1000]) {
      await page.setViewportSize({width:1440,height});
      await page.locator('#sidebar').evaluate(n => n.scrollTop = n.scrollHeight);
      const save = await page.locator('#mask-save').boundingBox(), sidebar = await page.locator('#sidebar').boundingBox();
      assert(save.y >= sidebar.y && save.y + save.height <= sidebar.y + sidebar.height, 'Save action stays within Paint pane');
    }
    await page.setViewportSize({width:390,height:844});
    const viewer = await page.locator('#workspace-main').boundingBox(), brush = await page.locator('#paint-quick-brush').boundingBox();
    assert(viewer.y < brush.y, 'Narrow Paint shows viewer before long tools');
    assert.equal(await page.locator('#tab-paint details').count(), 0);
    assert.deepEqual(errors, []);
    console.log('PASS import ranges, Run scope and checkpoint availability/recovery, real track pipeline, Inspect reasons and human review, pinned Paint saves and narrow viewer order');
  } catch (error) { console.error(logs); throw error; }
  finally { if (browser) await browser.close(); server.kill('SIGTERM'); await sleep(400); fs.rmSync(root, {recursive:true, force:true}); }
})();
