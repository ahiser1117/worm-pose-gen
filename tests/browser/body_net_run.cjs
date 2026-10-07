// Run from the repo root with Playwright + Chromium available (same env as mask_editor.cjs).
// Starts tests/browser/body_net_run_fixture.py and checks the Run panel's "Body-field network" checkbox
// (default on with the app's checkpoint, toggling, persistence, sync with the detailed fit-stage body_net,
// a custom path, the disabled state) and the params Run pipeline and a single stage Run send (job
// submissions are intercepted), plus the "fit with / without network" markers in the header and the list.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-body-net-run-'));
  const checkpoint = fs.realpathSync(root) + '/nets/best.ckpt';
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/body_net_run_fixture.py', '--root', root, '--port', String(port)], {
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
    const page = await browser.newPage({viewport: {width: 1400, height: 1000}}), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const text = selector => page.locator(selector).textContent();
    // Job submissions are recorded and reported done at once, so Run pipeline walks every included stage.
    const jobs = [];
    await page.route('**/api/jobs', async route => {
      const request = route.request();
      if (request.method() === 'POST') {
        const body = request.postDataJSON();
        const job = {id: `j${jobs.length + 1}`, state: 'done', progress: 1, finished_at: new Date().toISOString(),
          spec: {kind: 'stage', workspace: body.workspace, label: body.stage, params: {stage: body.stage, params: body.params}}, sent: body};
        jobs.push(job);
        return route.fulfill({contentType: 'application/json', body: JSON.stringify(job)});
      }
      return route.fulfill({contentType: 'application/json', body: JSON.stringify(jobs)});
    });
    await page.goto(`${base}/#ws=demo`);
    await page.waitForFunction(() => state.runName === 'demo' && state.stages && state.info?.body_net);

    // Markers: the header and the workspace list say how the current poses were fit.
    assert.equal(await text('#workspace-fit-network'), 'fit without network');
    const options = await page.locator('#run option').allTextContents();
    assert(options.some(o => /^networked .*· fit with network$/.test(o)), options.join(' | '));
    assert(options.some(o => /^demo .*· fit without network$/.test(o)), options.join(' | '));

    // Default: on with the app's checkpoint, which is also the detailed fit-stage value.
    await page.evaluate(() => showTab('rerun'));
    const toggle = page.locator('#workflow-body-net-toggle');
    await page.waitForFunction(() => document.querySelector('#workflow-body-net-toggle').checked);
    assert(await page.locator('#workflow-body-net').isVisible(), 'the checkbox is in the main Run section');
    assert.equal(await page.locator('#workflow-stage-details').evaluate(n => n.open), false, 'the detailed configuration stays collapsed');
    assert.match(await text('#workflow-body-net-status'), /^Network: best\.ckpt$/);
    assert.match(await text('#workflow-body-net'), /Scores fits with the network's A-P field and head\/tail and starts them from its traced midline; propagation and the track pass use it too\./);
    const detailed = page.locator('#stages .stage[data-stage="fit"] label.param').filter({has: page.locator('.param-name', {hasText: /^body_net$/})}).locator('input');
    assert.equal(await detailed.inputValue(), checkpoint);
    assert.match(await text('#workflow-run-summary'), /Current poses: fit without network/);
    await page.locator('#workflow-run-overview').screenshot({path: path.join(os.tmpdir(), 'worm-body-net-run-overview.png')});

    // Off is a stored choice: it survives a reload and empties the detailed field.
    await toggle.uncheck();
    assert.equal(await detailed.inputValue(), '');
    assert.match(await text('#workflow-body-net-status'), /^Off/);
    assert.deepEqual(await page.evaluate(() => JSON.parse(localStorage.getItem('poseViewer.stageParams')).fit), {body_net: null});
    await page.reload();
    // Wait for boot to finish: it restores the workspace's remembered view context, stage values included.
    await page.waitForFunction(() => state.runName === 'demo' && state.stages && state.info?.body_net && state.frame?.detail === 'full');
    await page.evaluate(() => showTab('rerun'));
    await page.waitForFunction(() => /^Off/.test(document.querySelector('#workflow-body-net-status').textContent));
    assert.equal(await toggle.isChecked(), false);

    // A path typed in the detailed field is a custom checkpoint; clearing it there turns the network off.
    await page.locator('#rerun-scope').selectOption('workspace');
    await page.locator('#workflow-stage-details').evaluate(n => n.open = true);
    await page.locator('#stages .stage[data-stage="fit"] .stage-toggle').click();
    await detailed.fill('/elsewhere/other.ckpt'); await detailed.press('Tab');
    await page.waitForFunction(() => document.querySelector('#workflow-body-net-toggle').checked);
    assert.equal(await text('#workflow-body-net-status'), 'custom checkpoint: other.ckpt');
    await detailed.fill(''); await detailed.press('Tab');
    await page.waitForFunction(() => !document.querySelector('#workflow-body-net-toggle').checked);
    assert.deepEqual(await page.evaluate(() => state.stageValues.fit), {body_net: null});
    await toggle.check();
    assert.equal(await detailed.inputValue(), checkpoint);

    // Run pipeline (segment left out: this app has no segmenter) sends the network to the fit stage only;
    // propagate and track read it from the fit summary.
    await page.locator('#stages .stage[data-stage="segment"] .stage-include').uncheck();
    await page.locator('#run-all').click();
    await page.waitForFunction(() => /Pipeline complete/.test(document.querySelector('#workflow-run-status').textContent), null, {timeout: 60000});
    const sent = Object.fromEntries(jobs.map(j => [j.sent.stage, j.sent.params]));
    assert.equal(sent.fit.body_net, checkpoint);
    for (const stage of ['propagate', 'track']) if (sent[stage]) assert(!('body_net' in sent[stage]), `${stage} does not take body_net`);
    assert(sent.propagate && sent.track, `stages sent: ${Object.keys(sent)}`);
    // A single fit Run sends it too, and so does an unchecked box as an explicit null.
    await toggle.uncheck();
    await page.locator('#stages .stage[data-stage="fit"] .stage-run').click();
    for (let i = 0; i < 50 && jobs.at(-1).sent.stage !== 'fit'; i++) await sleep(100);
    assert.deepEqual([jobs.at(-1).sent.stage, jobs.at(-1).sent.params.body_net], ['fit', null]);

    // Without the app's checkpoint the box is disabled and says why.
    await page.evaluate(() => { state.info.body_net = {path: '/nowhere/best.ckpt', exists: false}; delete state.stageValues.fit.body_net; workflowRun.render(); });
    assert.equal(await toggle.isDisabled(), true);
    assert.equal(await text('#workflow-body-net-status'), 'no body-field network at /nowhere/best.ckpt');

    // The other workspace's header marker.
    await page.evaluate(() => selectSource('workspace', 'networked'));
    await page.waitForFunction(() => state.runName === 'networked' && document.querySelector('#workspace-fit-network').textContent === 'fit with network');
    assert.match(await page.locator('#workspace-fit-network').getAttribute('title'), /best\.ckpt/);
    await page.screenshot({path: path.join(os.tmpdir(), 'worm-body-net-run.png')});
    assert.deepEqual(errors, []);
    console.log('body-field network: checkbox default, toggle, persistence, detailed-field sync, custom path, disabled state, Run pipeline and stage params, workspace markers passed');
  } finally {
    if (browser) await browser.close();
    server.kill();
    fs.rmSync(root, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
