// The Workspace page end to end on a synthetic CPU app (tests/browser/workspace_fixture.py):
// Recordings -> Analyse -> issues (and the left panel's layout with many of them) -> Looks OK ->
// Flip -> Refit (preview, Keep) -> Undo -> Edit mask (Save refits and keeps) -> Relabel round trip -> timeline scrub and
// right-drag selection -> Export.
// Run from the repository root: node tests/browser/workspace.cjs
// Environment: PLAYWRIGHT_MODULE, CHROMIUM_EXECUTABLE (and LD_LIBRARY_PATH for its libraries) when
// not installed in the default places; PYTHON (default .venv/bin/python); WS_SHOTS, a directory
// for screenshots of the main states (none are taken without it). Labeling's queue endpoints are
// stood in for with page.route: the queue answers {id}, and its stitch calls the workspace's own
// stitch endpoint with the keyframes' current poses, as Labeling's would with the labels'.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const WS = '2026-03-14-01';

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-workspace-e2e-'));
  const probe = net.createServer();
  await new Promise((resolve) => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise((resolve) => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/workspace_fixture.py', '--root', root, '--port', String(port)], {
    env: {...process.env, PYTHONPATH: `${process.cwd()}/src:${process.cwd()}`, CUDA_VISIBLE_DEVICES: '', OMP_NUM_THREADS: '4', MKL_NUM_THREADS: '4', MPLCONFIGDIR: path.join(root, 'mpl')},
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '', browser, failed = false;
  server.stdout.on('data', (b) => { logs = (logs + b).slice(-16000); });
  server.stderr.on('data', (b) => { logs = (logs + b).slice(-16000); });
  const shots = process.env.WS_SHOTS;
  if (shots) fs.mkdirSync(shots, {recursive: true});
  try {
    let ready = false;
    for (let i = 0; i < 120 && !ready && server.exitCode === null; i++) {
      try { ready = (await fetch(base + '/api/config')).ok; } catch { await sleep(500); }
    }
    assert.ok(ready, logs);
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport: {width: 1440, height: 900}}), errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('dialog', (dialog) => dialog.accept());
    const shot = async (name) => { if (shots) { await sleep(400); await page.screenshot({path: path.join(shots, `${name}.png`)}); } };
    const get = async (endpoint) => { const response = await page.request.get(base + endpoint); assert.equal(response.status(), 200, endpoint); return response.json(); };
    const fixCount = () => page.locator('.ws-fix-item').count();
    const issueSummary = async () => (await get(`/api/workspaces/${WS}/issues`)).summary;
    const waitFor = async (predicate, what, seconds = 600) => {
      for (let i = 0; i < seconds * 2; i++) { if (await predicate()) return; await sleep(500); }
      assert.fail(`timed out waiting for ${what}`);
    };

    // Recordings: both recordings of the setup, not analysed, with the setup's models.
    await page.goto(base + '/#workspace');
    await page.waitForSelector('tr[data-recording]');
    assert.equal(await page.locator('tr[data-recording]').count(), 2);
    assert.match(await page.locator('.ws-analyse-with').textContent(), /nir-hand284 \+ nir-body-lags3/);
    assert.equal(await page.locator('.ws-run-on').count(), 0, 'one place to run jobs: no Run on choice');
    await shot('01-recordings');
    // With a body model, Change offers the masks from it; the mask model stays the default.
    const masksFrom = async (value, label) => {
      await page.locator('.ws-analyse-with button:has-text("Change")').click();
      const source = page.locator('dialog.ws-analyse select[aria-label="Masks from"]');
      assert.ok(await source.isVisible(), 'a body model offers its masks');
      await source.selectOption(value);
      await page.locator('dialog.ws-analyse button:has-text("Use these models")').click();
      await page.waitForFunction((text) => document.querySelector('.ws-analyse-with strong')?.textContent === text, label);
    };
    await masksFrom('body_net', 'nir-body-lags3 (masks and body)');
    await masksFrom('segmenter', 'nir-hand284 + nir-body-lags3');

    // Nothing analysed yet: no "Analysed and analysing" table.
    assert.equal(await page.locator('.ws-section:not([hidden]) tr[data-recording]').count(), 2);
    // Analyse runs every default stage as one job and opens the workspace, with one bar per stage.
    await page.locator(`tr[data-recording="${WS}"] button:has-text("Analyse")`).click();
    await page.waitForFunction(() => location.hash === '#workspace/2026-03-14-01');
    await page.waitForSelector('.ws-analysing .ws-stage progress');
    const job = (await get('/api/jobs')).find((j) => j.spec.kind === 'analyse');
    assert.deepEqual(job.spec.params.stages, ['segment', 'prior', 'fit', 'ambiguity', 'propagate', 'track']);
    assert.equal(await page.locator('.ws-stage').count(), 6);
    assert.match(await page.locator('#header-context').textContent(), /2026-03-14-01.*nir-hand284/);
    await shot('02-analysing');
    // Back on Recordings while it runs: it heads the "Analysed and analysing" table, and the polls keep the scroll.
    await page.locator('.ws-back').click();
    await page.waitForSelector(`.ws-section:first-child tr[data-recording="${WS}"] .badge.info`);
    assert.equal(await page.locator('.ws-section:first-child tr[data-recording]').count(), 1);
    assert.equal(await page.locator(`tr[data-recording="${WS}"]`).count(), 1, 'each recording is listed once');
    await page.evaluate(() => { const lists = document.querySelector('.ws-lists'); lists.style.maxHeight = '120px'; lists.scrollTop = 60; });
    await sleep(3500);
    assert.equal(await page.evaluate(() => document.querySelector('.ws-lists').scrollTop), 60, 'a poll must not reset the scroll');
    await page.evaluate(() => { document.querySelector('.ws-lists').style.maxHeight = ''; });
    await page.locator(`.ws-section:first-child tr[data-recording="${WS}"] button:has-text("Open")`).click();
    await page.waitForSelector('.ws-analysing progress');
    await waitFor(async () => (await get(`/api/jobs/${job.id}`)).state !== 'running' && (await get(`/api/jobs/${job.id}`)).state !== 'queued', 'the analysis');
    assert.equal((await get(`/api/jobs/${job.id}`)).state, 'done', logs);
    await page.waitForSelector('.ws-issue', {timeout: 30000});
    await sleep(1500);
    await shot('03-issues');

    // A real recording has dozens of issues: the list scrolls inside its panel, and the fix panel (here the
    // mask tools) and the Fixes list stay in view in a smaller window, with no scrolling of the panel or page.
    const real = await get(`/api/workspaces/${WS}/issues`);
    const many = Array.from({length: 25}, (_, k) => ({...real.issues[k % real.issues.length], id: `many-${k}`}));
    const manyIssues = `**/api/workspaces/${WS}/issues`;
    await page.route(manyIssues, (route) => (route.request().method() === 'GET'
      ? route.fulfill({json: {...real, issues: many, summary: {...real.summary, issues: 25, done: 0}}}) : route.continue()));
    await page.setViewportSize({width: 1280, height: 720});
    await page.reload();
    await page.waitForFunction(() => document.querySelectorAll('.ws-issue').length === 25);
    await page.keyboard.press('e');
    await page.waitForSelector('.ws-mask-tools');
    const layout = await page.evaluate(() => {
      const box = (selector) => document.querySelector(selector).getBoundingClientRect();
      const left = document.querySelector('.ws-left'), list = document.querySelector('.ws-issue-list');
      return {
        page: document.scrollingElement.scrollHeight - window.innerHeight, panel: left.scrollHeight - left.clientHeight,
        listScrolls: list.scrollHeight > list.clientHeight + 1, leftBottom: box('.ws-left').bottom,
        save: box('.ws-mask-tools button.primary').bottom, fixes: box('.ws-fixes .ws-panel-head').bottom, nav: box('.ws-issue-nav').bottom,
      };
    });
    await shot('03b-many-issues');
    assert.ok(layout.page <= 0, `the page scrolls: ${JSON.stringify(layout)}`);
    assert.ok(layout.panel <= 1, `the left panel scrolls: ${JSON.stringify(layout)}`);
    assert.ok(layout.listScrolls, `the issue list does not scroll: ${JSON.stringify(layout)}`);
    for (const key of ['save', 'fixes', 'nav']) assert.ok(layout[key] <= layout.leftBottom, `${key} is out of view: ${JSON.stringify(layout)}`);
    await page.keyboard.press('Escape');
    await page.unroute(manyIssues);
    await page.setViewportSize({width: 1440, height: 900});
    await page.reload();
    await page.waitForSelector('.ws-issue', {timeout: 30000});
    await sleep(1000);

    // The first issue to review is selected; Looks OK reviews it and moves on.
    const before = await issueSummary();
    assert.ok(before.issues >= 1, JSON.stringify(before));
    assert.equal(await page.locator('.ws-issue[aria-current="true"]').count(), 1);
    await page.keyboard.press('o');
    await waitFor(async () => (await issueSummary()).done === before.done + 1, 'Looks OK', 30);
    assert.match(await page.locator('.ws-issues .ws-count').textContent(), new RegExp(`${before.done + 1} of ${before.issues} done`));

    // Flip the whole issue (F) and one frame (button); both land in the Fixes list.
    await page.locator('.ws-issue').first().click();
    await page.keyboard.press('f');
    await waitFor(async () => (await fixCount()) === 1, 'the flip', 30);
    await page.locator('.ws-fix button:has-text("Flip this frame")').click();
    await waitFor(async () => (await fixCount()) === 2, 'the frame flip', 30);
    assert.match(await page.locator('.ws-fix-item').first().textContent(), /Flipped head\/tail/);

    // Refit shows a before/after preview; Keep installs it.
    await page.locator('.ws-fix button:has-text("Refit")').click();
    await page.waitForSelector('.ws-preview', {timeout: 600000});
    assert.equal(await page.locator('.ws-legend').isVisible(), true);
    await shot('04-refit-preview');
    await page.locator('.ws-preview button:has-text("Keep")').click();
    await waitFor(async () => (await fixCount()) === 3, 'Keep', 30);
    assert.match(await page.locator('.ws-fix-item').first().textContent(), /Refit/);
    assert.equal((await get(`/api/workspaces/${WS}/fixes/previews`)).length, 0);
    await shot('05-kept');

    // Undo the newest fix.
    await page.locator('.ws-fix-item').first().locator('button:has-text("Undo")').click();
    await waitFor(async () => (await fixCount()) === 2, 'Undo', 30);
    assert.equal((await get(`/api/workspaces/${WS}/fixes`)).count, 2);

    // Edit mask: paint, Save & refit; the refit keeps itself.
    await page.keyboard.press('e');
    await page.waitForSelector('.ws-mask-tools');
    const box = await page.locator('.ws-canvas canvas').boundingBox();
    await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.25);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.52, box.y + box.height * 0.27, {steps: 6});
    await page.mouse.up();
    await shot('06-edit-mask');
    await page.locator('.ws-mask-tools button:has-text("Save & refit")').click();
    await waitFor(async () => (await get(`/api/workspaces/${WS}/fixes`)).fixes.some((f) => f.kind === 'refit'), 'the refit after the mask edit');
    await waitFor(async () => (await fixCount()) === 4, 'the mask edit and its refit in the list', 30);
    const kinds = (await get(`/api/workspaces/${WS}/fixes`)).fixes.map((f) => f.kind);
    assert.deepEqual(kinds.slice(0, 2), ['refit', 'mask']);

    // Relabel: keyframes on the timeline, a queue in Labeling, and back to stitch them.
    let queued = null;
    await page.route('**/api/queues', async (route) => {
      queued = route.request().postDataJSON();
      await route.fulfill({json: {id: 'q-test'}});
    });
    await page.route('**/api/queues/q-test/stitch', async (route) => {
      const keyframes = [];
      for (const frame of queued.frames) {
        const {pose} = await get(`/api/workspaces/${WS}/frame?frame=${frame}&detail=light`);
        keyframes.push({frame, centerline_xy: pose.centerline_xy, width_profile: pose.width_profile});
      }
      const response = await page.request.post(`${base}/api/workspaces/${WS}/fixes/stitch`, {data: {keyframes}});
      await route.fulfill({status: response.status(), json: await response.json()});
    });
    await page.locator('.ws-issue').first().click();
    await page.keyboard.press('l');
    await page.waitForSelector('.ws-relabel');
    const proposed = await page.evaluate(() => document.querySelector('.ws-relabel button.primary').textContent);
    await shot('07-relabel');
    await page.locator('.ws-relabel button.primary').click();
    await page.waitForURL(/#labeling\/queue\/q-test$/);
    assert.equal(queued.kind, 'relabel');
    assert.equal(queued.workspace, WS);
    assert.match(proposed, new RegExp(`Label ${queued.frames.length} keyframe`));
    // Labeling cannot open the stand-in queue, so come back on a fresh load, as "Back to workspace" lands.
    await page.goto('about:blank');
    await page.goto(`${base}/#workspace/${WS}/stitch/q-test`);
    await page.waitForSelector('.ws-preview', {timeout: 600000});
    assert.match(await page.locator('.ws-fix .ws-panel-head').textContent(), /Relabel/);
    assert.equal(page.url(), `${base}/#workspace/${WS}`);
    await page.locator('.ws-preview button:has-text("Discard")').click();
    await page.waitForSelector('.ws-fix-grid');

    // The frame fits the view only through the Fit button or 0, not a double-click.
    const frameBox = await page.locator('.ws-canvas canvas').boundingBox();
    const centre = [frameBox.x + frameBox.width / 2, frameBox.y + frameBox.height / 2];
    const picture = () => sleep(300).then(() => page.evaluate(() => {
      const c = document.querySelector('.ws-canvas canvas'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let hash = 0;
      for (let i = 0; i < d.length; i += 97) hash = (hash * 31 + d[i]) >>> 0;
      return hash;
    }));
    await sleep(1000);
    const fitted = await picture();
    await page.mouse.move(...centre);
    await page.mouse.wheel(0, -400);
    const zoomed = await picture();
    assert.notEqual(zoomed, fitted);
    await page.mouse.dblclick(...centre);
    assert.equal(await picture(), zoomed, 'a double-click does not fit the frame');
    await page.keyboard.press('0');
    assert.equal(await picture(), fitted, '0 fits the frame');
    await page.mouse.wheel(0, -400);
    await page.locator('.ws-fit').click();
    assert.equal(await picture(), fitted, 'the Fit button fits the frame');

    // The timeline: a left drag scrubs the frame as it goes, a right drag selects a range (Esc clears it).
    await page.keyboard.press('Escape');
    const timeline = await page.locator('.ws-timeline-canvas').boundingBox(), ty = timeline.y + 40;
    const along = (fraction) => timeline.x + timeline.width * fraction;
    await page.mouse.dblclick(along(0.05), ty);  // the whole recording in view
    await page.mouse.move(along(0.1), ty);
    await page.mouse.down();
    await page.mouse.move(along(0.6), ty, {steps: 8});
    await page.waitForFunction(() => Math.abs(Number(document.querySelector('.ws-frame-input').value) - 24) <= 1);
    await page.mouse.up();
    assert.equal((await page.locator('.ws-selection').textContent()).trim(), '', 'a left drag selects nothing');
    const scrubbed = await page.locator('.ws-frame-input').inputValue();
    await page.mouse.move(along(0.2), ty);
    await page.mouse.down({button: 'right'});
    await page.mouse.move(along(0.5), ty, {steps: 8});
    await page.mouse.up({button: 'right'});
    assert.match(await page.locator('.ws-selection').textContent(), /Selected 8–20/);
    assert.equal(await page.locator('.ws-frame-input').inputValue(), scrubbed, 'a right drag does not seek');
    await page.keyboard.press('Escape');
    assert.equal((await page.locator('.ws-selection').textContent()).trim(), '');

    // Export: unreviewed issues are named, the setup's pixel size and rate go along, the link downloads.
    await page.locator('#header-actions button:has-text("Export")').click();
    await page.waitForSelector('.ws-export');
    await shot('08-export');
    await page.locator('.ws-export button.primary').click();
    await page.waitForSelector('.ws-export-result a');
    const exports = await get(`/api/workspaces/${WS}/exports`);
    assert.equal(exports.length, 1);
    assert.equal(exports[0].pixel_size_um, 2.5);
    const download = await page.request.get(base + exports[0].download_url);
    assert.equal(download.status(), 200);
    await page.keyboard.press('Escape');

    // Back on the Recordings screen the row shows its state and when it was opened.
    await page.locator('.ws-back').click();
    await page.waitForSelector(`tr[data-recording="${WS}"] button:has-text("Open")`);
    assert.notEqual((await page.locator(`tr[data-recording="${WS}"] td`).nth(4).textContent()).trim(), '—');
    await shot('09-recordings-after');
    assert.deepEqual(errors, []);
    console.log('workspace.cjs: ok');
  } catch (error) {
    failed = true;
    console.error(error);
    console.error(logs.slice(-4000));
  } finally {
    if (browser) await browser.close();
    server.kill();
    fs.rmSync(root, {recursive: true, force: true});
    process.exitCode = failed ? 1 : 0;
  }
})();
