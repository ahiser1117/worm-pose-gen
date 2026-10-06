// Run from the repo root with Playwright + Chromium available (same env as mask_editor.cjs).
// Starts tests/browser/paint_fixture.py (a synthetic CPU app) and drives Paint as its own section:
// the label-group chooser, a manifest queue (first unlabeled entry, progress, next/previous, saves
// with split pledges, draft guard), a section sent from a workspace's Inspect selection, the Labels
// hand-off with return, and the workspace Masks task still editing workspace overrides.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-paint-'));
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/paint_fixture.py', '--root', root, '--port', String(port)], {
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
    const text = selector => page.locator(selector).textContent();
    const corpus = async () => (await (await fetch(base + '/api/corpus')).json()).samples;
    await page.goto(base);
    await page.waitForFunction(() => state.runName === 'demo');
    assert.deepEqual(await page.locator('#tabs button').allTextContents(), ['Run', 'Inspect', 'Masks', 'Compare', 'Export']);

    // Chooser: repository manifests are offered; a manifest path opens a queue on its first unlabeled entry.
    await page.locator('[data-main-tab="paint"]').click();
    await page.waitForFunction(() => document.querySelectorAll('#paint-manifests .paint-group').length >= 2);
    assert.match(await text('#paint-manifests'), /labeling_round_3_contact/);
    assert(await page.locator('#paint-workbench').isHidden());
    await page.locator('#paint-manifest-path').fill(path.join(root, 'queue.json'));
    await page.locator('#paint-manifest-open').click();
    await page.waitForFunction(() => /Entry 2 of 3/.test(document.querySelector('#paint-position').textContent) && paintScreen.current());
    assert.match(await text('#paint-group-name'), /Manifest · fixture queue/);
    assert.match(await text('#paint-progress'), /1 labeled · 2 remaining · 3 total/);
    assert.match(await text('#paint-entry'), /recording\.h5 · frame 3 · unlabeled · pledged val · self_contact/);
    assert.equal(await page.evaluate(() => state.screen), 'paint');

    // Paint a stroke on the Paint canvas, undo it, apply a proposal, save with S.
    const box = await page.locator('#paint-canvas').boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 40, box.y + box.height / 2, {steps: 5}); await page.mouse.up();
    await page.waitForFunction(() => paintScreen.isDirty());
    await page.keyboard.press('z');
    assert.equal(await page.evaluate(() => paintScreen.isDirty()), false);
    await page.keyboard.press('c');
    await page.waitForFunction(() => /classical preview · Apply/.test(document.querySelector('#paint-proposal-status').textContent));
    await page.keyboard.press('a');
    assert.equal(await page.evaluate(() => paintScreen.isDirty()), true);
    await page.keyboard.press('s');
    await page.waitForFunction(() => /2 labeled · 1 remaining/.test(document.querySelector('#paint-progress').textContent));
    const saved = (await corpus()).find(s => s.frame_index === 3);
    assert.equal(saved.split, 'val', 'the manifest pledge decides the split');

    // Next / previous walk the queue; a dirty draft asks before leaving its entry.
    await page.keyboard.press('n');
    await page.waitForFunction(() => /Entry 3 of 3/.test(document.querySelector('#paint-position').textContent) && paintScreen.current()?.frame === 5);
    await page.mouse.move(box.x + 50, box.y + 50); await page.mouse.down(); await page.mouse.move(box.x + 90, box.y + 60); await page.mouse.up();
    await page.keyboard.press('p');
    await page.waitForSelector('#draft-decision[open]');
    await page.locator('#draft-decision button[value="stay"]').click();
    assert.match(await text('#paint-position'), /Entry 3 of 3/);
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => /3 labeled · 0 remaining/.test(document.querySelector('#paint-progress').textContent));
    assert.match(await text('#paint-status'), /Every entry of this group is labeled/);
    await page.keyboard.press('p');
    await page.waitForFunction(() => /Entry 2 of 3/.test(document.querySelector('#paint-position').textContent));
    await page.locator('#paint-back').click();
    await page.waitForFunction(() => /3 labeled · 0 remaining/.test(document.querySelector('#paint-groups').textContent));

    // Inspect sends the workspace selection to Paint as a recording section; the workspace is not needed afterwards.
    await page.locator('#return-workspace').click();
    await page.locator('#tabs [data-tab="inspect"]').click();
    await page.locator('#selection-first').fill('2'); await page.locator('#selection-first').dispatchEvent('change');
    await page.locator('#selection-last').fill('4'); await page.locator('#selection-last').dispatchEvent('change');
    await page.waitForFunction(() => !document.querySelector('#inspection-label').disabled);
    await page.locator('#inspection-label').click();
    await page.waitForFunction(() => state.screen === 'paint' && /Recording section · demo frames 2-4/.test(document.querySelector('#paint-group-name').textContent) && paintScreen.current());
    assert.match(await text('#paint-position'), /Entry 1 of 3/, 'frames 2-4: frame 2 is the first unlabeled');
    const groups = await (await fetch(base + '/api/labeling/groups')).json();
    assert(groups.groups.some(g => g.kind === 'section' && g.origin.workspace === 'demo'));
    assert(fs.existsSync(path.join(root, 'workspaces', 'label_sections.json')));

    // Labels opens a saved label in Paint and returns.
    await page.locator('[data-main-tab="labels"]').click();
    await page.waitForFunction(() => document.querySelectorAll('#corpus-list .item').length === 3);
    await page.locator('#corpus-list .item').first().getByRole('button', {name: 'Open'}).click();
    await page.waitForFunction(() => state.screen === 'paint' && /Saved labels/.test(document.querySelector('#paint-group-name').textContent) && paintScreen.current());
    assert.equal(await text('#paint-back'), 'Return to Labels');
    await page.locator('#paint-back').click();
    assert.equal(await page.evaluate(() => state.screen), 'labels');

    // The workspace Masks task still edits the workspace override that fitting reads.
    await page.locator('#return-workspace').click();
    await page.locator('#tabs [data-tab="masks"]').click();
    await page.waitForFunction(() => /Workspace mask · frame/.test(document.querySelector('#mask-target').textContent) && !document.querySelector('#mask-save').disabled);
    const stage = await page.locator('#canvas').boundingBox();
    await page.mouse.move(stage.x + stage.width / 2, stage.y + stage.height / 2); await page.mouse.down();
    await page.mouse.move(stage.x + stage.width / 2 + 30, stage.y + stage.height / 2, {steps: 4}); await page.mouse.up();
    assert.equal(await page.evaluate(() => maskEditor.isDirty()), true);
    await page.keyboard.press('s');
    await page.waitForFunction(() => /Mask saved/.test(document.querySelector('#mask-status').textContent));
    const frame = await page.evaluate(() => state.run.series.frame_index[state.row]);
    const mask = await (await fetch(`${base}/api/workspaces/demo/mask?frame=${frame}`)).json();
    assert.equal(mask.has_override, true);
    await page.screenshot({path: path.join(os.tmpdir(), 'worm-paint-masks.png')});
    assert.deepEqual(errors, []);
    console.log('paint section: chooser, manifest queue (first unlabeled, progress, next/previous, draft guard, pledged saves), Inspect section, Labels hand-off, workspace Masks passed');
  } finally {
    if (browser) await browser.close();
    server.kill();
    fs.rmSync(root, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
