// Synthetic Import/Open/Labels/Training checks. No production data is touched.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-usability-library-'));
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/phase4_fixture.py', '--root', root, '--port', String(port)], {
    env: {...process.env, PYTHONPATH: `${process.cwd()}/src:${process.cwd()}`, CUDA_VISIBLE_DEVICES: '', OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1', MPLCONFIGDIR: path.join(root, 'mpl')}, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '', browser;
  server.stdout.on('data', b => logs = (logs + b).slice(-16000));
  server.stderr.on('data', b => logs = (logs + b).slice(-16000));
  try {
    let ready = false;
    for (let i = 0; i < 60; i++) {
      try { ready = (await fetch(base + '/api/state')).ok; } catch {}
      if (ready || server.exitCode !== null) break;
      await sleep(500);
    }
    assert.ok(ready, logs);
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}}), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.waitForFunction(() => state.runName === 'demo');
    await page.locator('#app-header [data-screen="import"]').click();
    const recording = page.getByRole('button', {name: 'Select recording recording', exact: true});
    await recording.focus();
    await recording.press('Enter');
    await page.waitForFunction(() => state.recording?.name === 'recording');
    assert.equal(await recording.getAttribute('aria-pressed'), 'true');
    assert.equal(await page.evaluate(() => document.activeElement.className), 'recording-select');
    const browserBox = await page.locator('.import-columns > .group').first().boundingBox();
    const previewBox = await page.locator('.import-preview').boundingBox();
    assert(previewBox.x > browserBox.x + browserBox.width, 'Preview sits beside the recording browser');
    const createBox = await page.locator('#ws-create').boundingBox();
    assert(createBox.y + createBox.height <= 1000, 'Create workspace is visible without scrolling');
    assert.equal(await page.locator('#ws-range-mode').inputValue(), 'entire');
    await page.locator('#ws-range-mode').selectOption('selected');
    await page.locator('#ws-first').fill('1');
    await page.locator('#ws-last').fill('3');
    await page.locator('#ws-step').fill('2');
    assert.match(await page.locator('#ws-range-summary').textContent(), /2 frames/);
    await page.locator('#explorer-path').fill(root + '/missing-child');
    await page.locator('#explorer-go').click();
    await page.waitForFunction(() => document.querySelector('#explorer-path').getAttribute('aria-invalid') === 'true');
    assert(await page.locator('#explorer-retry').isVisible());
    await page.locator('#explorer-choose').click();
    assert.equal(await page.evaluate(() => document.activeElement.id), 'explorer-path');
    await page.locator('#explorer-up').click();
    await page.waitForFunction(expected => document.querySelector('#explorer-path').value === expected && !document.querySelector('#explorer-path').hasAttribute('aria-invalid'), root);
    // A pending preparation occupies the preview region; Close stays usable.
    await page.evaluate(() => { window.libraryProgress = beginPreviewTask('Synthetic preparation'); window.libraryProgress.update('prepare', 'Preparing illumination correction…'); });
    assert.equal(await page.locator('#import-preview-progress #preview-progress').count(), 1);
    assert.equal(await page.locator('#preview-progress').evaluate(node => getComputedStyle(node).position), 'static');
    assert(await page.locator('#screen-import [data-close-panel]').isVisible());
    await page.evaluate(() => window.libraryProgress.finish());
    await page.locator('#app-header [data-screen="open"]').click();
    await page.locator('#run').selectOption('ws:demo');
    assert(await page.locator('#screen-open').isVisible());
    assert.equal(await page.locator('#open-selected').textContent(), 'Resume workspace');
    assert(await page.locator('#import-run').isHidden());
    await page.locator('#open-selected').click();
    await page.waitForFunction(() => state.screen === 'workspace');
    await page.evaluate(() => showTab('training'));
    await page.waitForFunction(() => document.querySelector('#training-counts').textContent.includes('0 training'));
    assert(await page.locator('#corpus-train').isDisabled());
    assert.match(await page.locator('#training-readiness').textContent(), /training label and one validation label/);
    assert(await page.locator('#tune-epochs').isVisible());
    assert(await page.locator('#tune-crop').isHidden());
    await page.locator('.training-advanced summary').click();
    assert(await page.locator('#tune-crop').isVisible());
    await page.locator('#training-jobs').click();
    await page.waitForFunction(() => state.rightTab === 'jobs');
    assert(await page.locator('#right-jobs').isVisible());
    assert(await page.locator('#right-statistics').isHidden());
    await page.locator('#training-add-labels').click();
    assert(await page.locator('#screen-labels').isVisible());
    await page.locator('#corpus-filter').fill('no-match');
    await page.getByRole('button', {name: 'Clear filters', exact: true}).click();
    assert.equal(await page.locator('#corpus-filter').inputValue(), '');
    for (const screen of ['import', 'open', 'labels', 'training']) {
      await page.setViewportSize({width: 320, height: 900});
      await page.evaluate(screen => showTab(screen), screen);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${screen} horizontal overflow`);
    }
    assert.deepEqual(errors, []);
    console.log('PASS: keyboard recording selection, exact range, directory recovery, inline preparation, explicit resume, training prerequisites/disclosure, labels recovery, narrow layout');
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
    await new Promise(resolve => { if (server.exitCode !== null) resolve(); else server.once('exit', resolve); });
    fs.rmSync(root, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
