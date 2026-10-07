// Run from the repo root with Playwright + Chromium available (PLAYWRIGHT_MODULE, CHROMIUM_EXECUTABLE as for the other checks).
// Starts tests/browser/labeling_fixture.py (a synthetic CPU app; or uses one already running: LABELING_URL and
// LABELING_ROOT) and drives the Labeling page: open a Relabel queue, paint, a network proposal, use the body proposal,
// flip, Save & next, trace a midline, mask only, Back to workspace, then Browse labels sorted by body fit IoU.
// SCREENSHOTS=<dir> saves the main states.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

const WIDTH = 320, HEIGHT = 240;
// The fixture's worm on frame k: head at x=60, tail at x=260.
const midY = (x, k) => HEIGHT / 2 + 28 * Math.sin((x - 60) / 45 + k * 0.15);

(async () => {
  let base = process.env.LABELING_URL, root = process.env.LABELING_ROOT, server = null, logs = '';
  if (!base) {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-labeling-'));
    const probe = net.createServer();
    await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
    const port = probe.address().port;
    await new Promise(resolve => probe.close(resolve));
    base = `http://127.0.0.1:${port}`;
    server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/labeling_fixture.py', '--root', root, '--port', String(port)], {
      env: {...process.env, PYTHONPATH: `${process.cwd()}/src:${process.cwd()}`, CUDA_VISIBLE_DEVICES: '', MPLCONFIGDIR: path.join(root, 'mpl')},
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    server.stdout.on('data', b => { logs = (logs + b).slice(-16000); });
    server.stderr.on('data', b => { logs = (logs + b).slice(-16000); });
  }
  const shots = process.env.SCREENSHOTS;
  let browser;
  try {
    let ready = false;
    for (let i = 0; i < 120 && !ready && (!server || server.exitCode === null); i++) {
      try { ready = (await fetch(base + '/api/config')).ok && fs.existsSync(path.join(root, 'fixture.json')); } catch { /* not up yet */ }
      if (!ready) await sleep(500);
    }
    assert.ok(ready, logs);
    const {relabel} = JSON.parse(fs.readFileSync(path.join(root, 'fixture.json'), 'utf8'));
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport: {width: 1440, height: 900}}), errors = [], saves = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', dialog => dialog.accept());
    page.on('request', request => { if (request.url().endsWith('/api/labeling/save')) saves.push(request.postDataJSON()); });
    const shot = async name => { if (shots) await page.screenshot({path: path.join(shots, `${name}.png`)}); };
    const button = name => page.locator('#page-labeling').getByRole('button', {name, exact: true});
    const status = () => page.locator('.lb-status').textContent();
    const bodyNote = () => page.locator('.lb-body-note').textContent();
    // Screen position of image pixel (x, y) in the fitted view.
    const at = async (x, y) => {
      const box = await page.locator('.lb-canvas canvas').boundingBox();
      const scale = Math.min(box.width / WIDTH, box.height / HEIGHT) * 0.98;
      return {x: box.x + (box.width - WIDTH * scale) / 2 + x * scale, y: box.y + (box.height - HEIGHT * scale) / 2 + y * scale};
    };
    const drag = async (from, to) => {
      const a = await at(...from), b = await at(...to);
      await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move(b.x, b.y, {steps: 5}); await page.mouse.up();
    };
    const click = async (x, y) => { const p = await at(x, y); await page.mouse.click(p.x, p.y); };
    const waitProposal = () => page.waitForFunction(() => /Proposal ready/.test(document.querySelector('.lb-body-note')?.textContent || ''), null, {timeout: 180000});
    // The fixture's server has no GPU: the body proposal is computed only when asked (Propose).
    const propose = async () => {
      await page.waitForSelector('.lb-body-controls button:text-is("Propose"):visible');
      assert.match(await page.locator('.lb-body-note').textContent(), /Propose computes the model's body/);
      await button('Propose').click();
      await waitProposal();
      assert.equal(await button('Propose').isDisabled(), true);
    };

    // Home: the setup's queues, with the Relabel queue from the workspace.
    await page.goto(base + '/#labeling');
    await page.waitForSelector(`.lb-queue[data-queue="${relabel}"]`);
    assert.match(await page.locator('#header-context').textContent(), /Saving to mine:nir-labels \(new\)/);
    assert.match(await page.locator('.lb-queue').first().textContent(), /Relabel ws frames 10–30[\s\S]*0 of 3 saved/);
    await shot('labeling-home');

    // New queue: the setup's recordings and a number of frames (the search itself is covered by tests/test_queues.py).
    await button('New queue…').click();
    await page.waitForSelector('.lb-recordings input[type="checkbox"]');
    assert.match(await page.locator('.lb-recordings').textContent(), /2024-05-05-01[\s\S]*60 frames/);
    await shot('labeling-new-queue');
    await button('← Queues').click();
    await page.waitForSelector(`.lb-queue[data-queue="${relabel}"]`);

    // The queue opens on its first frame; the mask comes from the model (the workspace has none); Propose computes the body proposal.
    await page.locator(`.lb-queue[data-queue="${relabel}"]`).click();
    await page.waitForFunction(() => /frame 10/.test(document.querySelector('.lb-status')?.textContent || ''));
    assert.equal(await page.evaluate(() => location.hash), `#labeling/queue/${relabel}/0`);
    assert.match(await status(), /2024-05-05-01 · frame 10 · split train · mask from the model/);
    await propose();
    await shot('labeling-editor');

    // Paint worm in an empty corner: unsaved; Undo takes it back. A background stroke and a network proposal (A applies it).
    await drag([20, 20], [40, 30]);
    assert.match(await status(), /unsaved changes/);
    await page.keyboard.press('z');
    assert.doesNotMatch(await status(), /unsaved changes/);
    await page.keyboard.press('e');
    assert.equal(await button('Background').getAttribute('aria-pressed'), 'true');
    await drag([255, 100], [262, 140]);
    assert.match(await status(), /unsaved changes/);
    await page.keyboard.press('n');
    await page.waitForSelector('.lb-right .row:has(button:text-is("Apply (A)"))', {state: 'visible'});
    await page.locator('[aria-label="Threshold"]').fill('0.4');
    await page.keyboard.press('a');
    await page.waitForSelector('.lb-right .row:has(button:text-is("Apply (A)"))', {state: 'hidden'});
    await page.keyboard.press('w');

    // Use the proposal (G), flip it (H), Save & next (Enter).
    await page.keyboard.press('g');
    await page.waitForFunction(() => /Body: traced/.test(document.querySelector('.lb-body-note').textContent));
    await page.keyboard.press('h');
    await page.locator('.lb-canvas canvas').focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => /frame 20/.test(document.querySelector('.lb-status')?.textContent || ''), null, {timeout: 60000});
    let saved = saves.at(-1);
    assert.equal(saved.queue, relabel);
    assert.ok(saved.trace_xy[0][0] > 200, `flipped: the head is at the right end (${saved.trace_xy[0]})`);
    assert.equal(saved.mask_only, false);
    assert.match(await page.locator('.lb-left').textContent(), /1 of 3 saved/);
    assert.match(await page.locator('#header-context').textContent(), /Saving to mine:nir-labels/);
    assert.doesNotMatch(await page.locator('#header-context').textContent(), /\(new\)/);

    // Trace the midline by hand from head to tail: Fit, then Accept; S saves in place. No proposal was asked for on this frame.
    assert.match(await page.locator('.lb-body-note').textContent(), /Propose computes the model's body/);
    await page.keyboard.press('t');
    for (const x of [62, 110, 160, 210, 258]) await click(x, midY(x, 20));
    assert.match(await page.locator('.lb-trace').textContent(), /5 points/);
    await page.keyboard.press('Backspace');
    assert.match(await page.locator('.lb-trace').textContent(), /4 points/);
    await click(258, midY(258, 20));
    await button('Fit').click();
    await page.waitForSelector('.lb-trace button:text-is("Accept"):visible', {timeout: 120000});
    assert.match(await page.locator('.lb-trace').textContent(), /Fit IoU 0\.\d+/);
    await shot('labeling-trace');
    await button('Accept').click();
    assert.match(await bodyNote(), /Body: traced, fit IoU/);
    await page.keyboard.press('s');
    await page.waitForFunction(() => /Saved revision 1/.test(document.querySelector('.lb-save-note').textContent));
    saved = saves.at(-1);
    assert.equal(saved.trace_xy.length, 5);
    assert.ok(saved.trace_xy[0][0] < 70, 'the hand trace starts at the head');
    assert.match(await status(), /frame 20/);

    // Temporal context: Difference (D), back to Frames, an offset.
    await page.keyboard.press('d');
    await page.waitForFunction(() => document.querySelector('.lb-context button[aria-pressed="true"]')?.textContent === 'Difference', null, {timeout: 60000});
    await page.keyboard.press('d');
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('.lb-context .lb-value').textContent === 't+1');

    // The last keyframe is mask only; then the queue is done and offers Back to workspace.
    await page.keyboard.press('Shift+ArrowRight');
    await page.waitForFunction(() => /frame 30/.test(document.querySelector('.lb-status')?.textContent || ''));
    await page.locator('.lb-maskonly input').check();
    await button('Save & next').click();
    await page.waitForSelector('button:text-is("Back to workspace")');
    assert.equal(saves.at(-1).mask_only, true);
    assert.match(await page.locator('.lb-left').textContent(), /3 of 3 saved/);
    await shot('labeling-queue-done');
    await button('Back to workspace').click();
    await page.waitForFunction(() => location.hash.startsWith('#workspace/'));
    assert.equal(await page.evaluate(() => location.hash), `#workspace/ws/stitch/${relabel}`);
    const queue = await (await fetch(`${base}/api/queues/${relabel}`)).json();
    assert.deepEqual(queue.entries.map(e => e.saved?.dataset), ['mine:nir-labels', 'mine:nir-labels', 'mine:nir-labels']);

    // Browse labels: the lab's labels (inherited) and mine, lowest body fit IoU first, then the ones not built yet.
    await page.goto(base + '/#labeling/browse');
    await page.waitForFunction(() => document.querySelectorAll('.lb-entries .lb-entry').length === 7);
    const ious = await page.locator('.lb-entries .lb-iou').allTextContents();
    assert.deepEqual(ious.slice(0, 3), ['0.710', '0.880', '0.930']);
    assert.ok(ious.slice(3).every(v => v === '—'));
    assert.match(await page.locator('.lb-entries .lb-entry').first().innerText(), /frame 5\b[\s\S]*body unconfirmed · 2024-05-05-01/);
    await page.locator('.lb-left label:has-text("Self-contact only") input').check();
    await page.waitForFunction(() => document.querySelectorAll('.lb-entries .lb-entry').length === 1);
    assert.match(await page.locator('.lb-entries .lb-entry').innerText(), /frame 25\b/);
    await page.locator('.lb-left label:has-text("Self-contact only") input').uncheck();
    await page.waitForFunction(() => document.querySelectorAll('.lb-entries .lb-entry').length === 7);
    await page.locator('[aria-label="Status"]').selectOption('mask_only');
    await page.waitForFunction(() => document.querySelectorAll('.lb-entries .lb-entry').length === 1);
    assert.match(await page.locator('.lb-entries .lb-entry').textContent(), /frame 30[\s\S]*mask only/);
    await page.locator('[aria-label="Status"]').selectOption('');
    await page.waitForFunction(() => document.querySelectorAll('.lb-entries .lb-entry').length === 7);
    await page.locator('.lb-entries .lb-entry').first().click();
    await page.waitForFunction(() => /frame 5 · split train · saved revision 1 \(lab:nir-labels\)/.test(document.querySelector('.lb-status')?.textContent || ''));
    assert.equal(await page.locator('.lb-maskonly input').isChecked(), false);
    await propose();
    await button('A-P').click();
    assert.equal(await button('A-P').getAttribute('aria-pressed'), 'true');
    await shot('labeling-browse');

    // Leaving a frame with unsaved changes asks first (the dialog is accepted here).
    await drag([20, 20], [40, 30]);
    await page.keyboard.press('Shift+ArrowRight');
    await page.waitForFunction(() => /frame 25/.test(document.querySelector('.lb-status')?.textContent || ''));

    // With a GPU for the server's models the proposal comes with the frame, and there is no Propose button.
    await page.route('**/api/config', async (route) => {
      const response = await route.fetch();
      await route.fulfill({response, json: {...(await response.json()), gpu: true}});
    });
    await page.goto(base + '/#labeling/browse');
    await page.reload();
    await page.waitForFunction(() => document.querySelectorAll('.lb-entries .lb-entry').length === 7);
    await page.locator('.lb-entries .lb-entry').first().click();
    await waitProposal();
    assert.equal(await button('Propose').isHidden(), true);
    await page.unroute('**/api/config');

    // The Training page's Datasets tab links one dataset's recording.
    await page.goto(base + '/#labeling/browse/lab:nir-labels/2024-05-05-01');
    await page.waitForFunction(() => document.querySelectorAll('.lb-entries .lb-entry').length === 4);
    assert.match(await page.locator('.lb-dataset').textContent(), /Labels of lab:nir-labels/);
    assert.equal(await page.locator('[aria-label="Recording"]').inputValue(), '2024-05-05-01');

    // The help lists the shortcuts.
    await page.keyboard.press('?');
    await page.waitForSelector('dialog.lb-help[open]');
    assert.match(await page.locator('dialog.lb-help').textContent(), /Use the body proposal/);
    await page.keyboard.press('Escape');

    assert.deepEqual(errors, []);
    console.log('labeling: ok');
  } finally {
    if (browser) await browser.close();
    if (server) server.kill();
  }
})().catch(error => { console.error(error); process.exit(1); });
