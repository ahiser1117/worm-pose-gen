// The Training page and the model picker against tests/browser/training_fixture.py (synthetic libraries, CPU jobs).
// SCREENSHOT_DIR, when set, receives one PNG per tab and of the picker and the details.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-training-'));
  const shots = process.env.SCREENSHOT_DIR;
  if (shots) fs.mkdirSync(shots, {recursive: true});
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(process.env.PYTHON || '.venv/bin/python', ['tests/browser/training_fixture.py', '--root', root, '--port', String(port)], {
    env: {...process.env, PYTHONPATH: `${process.cwd()}/src:${process.cwd()}`, CUDA_VISIBLE_DEVICES: '', OMP_NUM_THREADS: '2', MKL_NUM_THREADS: '2', MPLCONFIGDIR: path.join(root, 'mpl')},
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '', browser;
  server.stdout.on('data', b => logs = (logs + b).slice(-16000));
  server.stderr.on('data', b => logs = (logs + b).slice(-16000));
  const shot = async (page, name) => { if (shots) await page.screenshot({path: path.join(shots, `${name}.png`)}); };
  try {
    let ready = false;
    for (let i = 0; i < 240 && !ready && server.exitCode === null; i++) {
      try { ready = (await fetch(base + '/api/config')).ok; } catch {}
      if (!ready) await sleep(500);
    }
    assert.ok(ready, logs);
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport: {width: 1440, height: 900}}), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });

    // ----- Models: the table, the running and the failed run.
    await page.goto(base + '/#training/models');
    const rows = page.locator('.mp-table tbody tr');
    await page.waitForFunction(() => document.querySelectorAll('.mp-table tbody tr').length === 3);
    const row = ref => page.locator(`.mp-table tr[data-ref="${ref}"]`);
    assert.match(await row('lab:body').innerText(), /★ body/);
    assert.match(await row('lab:seg').innerText(), /★ mask/);
    assert.match(await row('lab:seg').innerText(), /no head\/tail: orientation from body taper only/);
    assert.match(await row('lab:body').innerText(), /frame \+ motion ±1\/2 fr \(0\.05\/0\.1 s\)/);
    assert.match(await row('mine:copper-ft').innerText(), /Mine/);
    assert.match(await row('mine:copper-ft').innerText(), /from body/);
    assert.match(await page.locator('.mp-wrap').innerText(), /lab:rig-v1 has 1 label: its worst 5% is the worst 1 label/);
    // The running job's card has a live curve and Cancel; the failed one its error and Dismiss.
    const running = page.locator('.tr-run:not(.failed)');
    await running.locator('polyline.tr-val').waitFor();
    assert.match(await running.innerText(), /copper-ft-2/);
    assert.match(await running.innerText(), /epoch \d+ of at most 40/);
    const failed = page.locator('.tr-run.failed');
    assert.match(await failed.innerText(), /CUDA out of memory/);
    // lab:seg was never evaluated: its evaluation is queued on its own and its numbers arrive.
    await page.waitForFunction(() => /\d\.\d{3}/.test(document.querySelector('.mp-table tr[data-ref="lab:seg"]')?.innerText || ""), null, {timeout: 120000});
    assert.equal(await page.locator('.mp-table tr[data-ref="lab:seg"] td.mp-pending').count(), 0);
    await page.waitForFunction(() => document.querySelectorAll('.tr-run:not(.failed) polyline.tr-val').length && /epoch 6/.test(document.querySelector('.tr-run:not(.failed)').innerText));
    await shot(page, 'models');
    await failed.getByRole('button', {name: 'Dismiss'}).click();
    await page.waitForFunction(() => !document.querySelector('.tr-run.failed'));
    await running.getByRole('button', {name: 'Cancel'}).click();
    await page.waitForFunction(() => !document.querySelector('.tr-run'));

    // ----- Details of the personal fine-tune.
    await row('mine:copper-ft').getByRole('button', {name: 'Details'}).click();
    const details = page.locator('dialog.tr-details');
    await details.locator('polyline.tr-val').waitFor();
    assert.match(await details.innerText(), /Started from\s+lab:body/);
    assert.match(await details.innerText(), /heads looked better on copper-1/);
    assert.match(await details.innerText(), /copper-1\s+2\s+0/);
    await details.locator('.tr-worst img').first().waitFor();
    await page.waitForFunction(() => document.querySelector('dialog.tr-details .tr-worst img').naturalWidth > 0);
    await shot(page, 'details');
    await details.getByRole('button', {name: 'Close'}).click();

    // ----- Use as default: the fine-tune becomes the body default, with a reason.
    await row('mine:copper-ft').getByRole('button', {name: 'Use as default'}).click();
    const dialog = page.locator('dialog.tr-dialog');
    await dialog.getByRole('button', {name: 'Use as default'}).click();
    assert.match(await dialog.innerText(), /one-line reason/);
    await dialog.getByRole('textbox', {name: 'Reason'}).fill('better heads on copper');
    assert.equal(await dialog.locator('input[value="mask"]').isChecked(), true);  // not yet the mask default
    await dialog.locator('input[value="body"]').check();
    await dialog.getByRole('button', {name: 'Use as default'}).click();
    await page.waitForFunction(() => /★ body/.test(document.querySelector('.mp-table tr[data-ref="mine:copper-ft"]')?.innerText || ""));
    assert.doesNotMatch(await row('lab:body').innerText(), /★ body/);
    const setup = await (await fetch(base + '/api/library/setups/lab:rig')).json();
    assert.equal(setup.defaults.body, 'mine:copper-ft');
    assert.equal(setup.defaults_log.at(-1).reason, 'better heads on copper');

    // ----- Datasets: the collection's recordings with a split each, live totals, New dataset, Browse, Freeze benchmark.
    await page.locator('.tr-tabs [data-tab="datasets"]').click();
    const totals = page.locator('.tr-split-totals');
    const tile = (split) => totals.locator(`.tr-split-${split} strong`);
    const splitOf = (recording) => page.locator(`tr[data-recording="${recording}"] select`);
    await totals.waitFor();
    assert.equal(await page.locator('.tr-dataset-select').inputValue(), 'mine:copper');  // a personal dataset first
    assert.deepEqual([await tile('train').innerText(), await tile('val').innerText(), await tile('test').innerText(), await tile('none').innerText()],
      ['4', '1', '2', '0']);
    assert.match(await totals.innerText(), /only 4 training labels/);
    assert.equal(await splitOf('copper-2').inputValue(), 'test');
    // Taking the validation recording out shows at once and is saved.
    await splitOf('rec-val').selectOption('');
    await page.waitForFunction(() => document.querySelector('.tr-split-totals .tr-split-val strong').textContent === '0');
    assert.match(await totals.innerText(), /no validation recording/);
    await page.waitForFunction(() => !document.querySelector('tr[data-recording="rec-val"] select').disabled);
    let copper = await (await fetch(base + '/api/library/datasets/mine:copper')).json();
    assert.deepEqual([copper.by_split, copper.not_included], [{train: 4, val: 0, test: 2}, {recordings: 1, labels: 1}]);
    await splitOf('rec-val').selectOption('val');
    await page.waitForFunction(() => document.querySelector('.tr-split-totals .tr-split-val strong').textContent === '1');
    await page.waitForFunction(() => !document.querySelector('tr[data-recording="rec-val"] select').disabled);
    await shot(page, 'datasets');
    // A new dataset includes nothing until its recordings get a split.
    await page.getByRole('button', {name: 'New dataset…'}).click();
    const make = page.locator('dialog.tr-dialog');
    await make.getByRole('textbox', {name: 'Name'}).fill('Heat shock');
    await make.getByRole('button', {name: 'Create'}).click();
    await page.waitForFunction(() => location.hash === '#training/datasets/mine%3Aheat-shock');
    await page.waitForFunction(() => document.querySelector('.tr-dataset-select')?.value === 'mine:heat-shock');
    assert.deepEqual(await page.locator('.tr-recordings select').evaluateAll(s => s.map(x => x.value)), ['', '', '', '', '']);
    assert.equal(await tile('none').innerText(), '7');
    await splitOf('copper-1').selectOption('train');
    await page.waitForFunction(() => document.querySelector('.tr-split-totals .tr-split-train strong').textContent === '2');
    await page.waitForFunction(() => !document.querySelector('tr[data-recording="copper-1"] select').disabled);
    const heat = await (await fetch(base + '/api/library/datasets/mine:heat-shock')).json();
    assert.deepEqual(heat.by_split, {train: 2, val: 0, test: 0});
    // A lab dataset shows its splits read-only.
    await page.locator('.tr-dataset-select').selectOption('lab:base');
    await page.waitForFunction(() => location.hash === '#training/datasets/lab%3Abase');
    await page.locator('.tr-recordings').waitFor();
    assert.equal(await page.locator('.tr-recordings select').count(), 0);
    await page.locator('.tr-dataset-select').selectOption('mine:copper');
    await page.waitForFunction(() => document.querySelector('.tr-dataset-select')?.value === 'mine:copper' && document.querySelector('.tr-recordings select'));
    await page.locator('tr[data-recording="copper-1"]').getByRole('button', {name: 'copper-1'}).click();
    await page.waitForFunction(() => location.hash === '#labeling/browse/mine%3Acopper/copper-1');
    await page.goto(base + '/#training/datasets/mine%3Acopper');
    await page.getByRole('button', {name: 'Freeze benchmark'}).click();
    await page.locator('dialog.tr-dialog').getByRole('button', {name: 'Freeze benchmark'}).click();
    await page.waitForFunction(() => /Froze mine:copper-b1/.test(document.querySelector('#toast').textContent));
    const benchmarks = await (await fetch(base + '/api/library/benchmarks?setup=lab:rig')).json();
    assert.deepEqual(benchmarks.benchmarks.map(b => b.ref).sort(), ['lab:rig-v1', 'mine:copper-b1']);

    // ----- Train: the form, then a real one-epoch fine-tune on the CPU.
    await page.locator('.tr-tabs [data-tab="train"]').click();
    const form = page.locator('form.tr-form');
    await form.waitFor();
    assert.equal(await form.locator('select[name="start_from"]').inputValue(), 'mine:copper-ft');  // the setup's body default
    assert(await page.getByText('Temporal context').isHidden());
    await page.waitForFunction(() => /training and \d+ validation labels/.test(document.querySelector('.tr-plan').textContent));
    assert.match(await form.locator('.tr-plan').innerText(), /4 training and 1 validation labels from 3 recordings/);
    assert.equal(await form.locator('input[name="name"]').inputValue(), 'copper-4');
    assert.equal(await form.locator('select[name="dataset"]').inputValue(), 'mine:copper');
    assert(await form.locator('input[name="crop_size"]').isHidden());
    await form.locator('select[name="start_from"]').selectOption('');
    assert(await page.getByText('Temporal context').isVisible());
    await form.locator('select[name="start_from"]').selectOption('lab:seg');
    assert.match(await form.locator('.tr-kind').innerText(), /Fine-tunes the mask-only segmenter/);
    assert.equal(await form.locator('input[name="patience"]').inputValue(), '5');
    await shot(page, 'train');
    await form.locator('select[name="start_from"]').selectOption('mine:copper-ft');
    await form.locator('input[name="max_epochs"]').fill('1');
    await form.locator('summary').click();
    await form.locator('input[name="crop_size"]').fill('64');
    await form.locator('input[name="name"]').fill('copper-quick');
    await form.locator('textarea[name="notes"]').fill('one epoch from the browser test');
    await page.waitForFunction(() => !document.querySelector('form.tr-form button[type="submit"]').disabled);
    await form.getByRole('button', {name: 'Start training'}).click();
    await page.waitForFunction(() => location.hash === '#training/models');
    await page.waitForFunction(() => document.querySelector('.mp-table tr[data-ref="mine:copper-quick"]'), null, {timeout: 240000});
    // The card is written before the job evaluates it; the run's card leaves when the job is done.
    await page.waitForFunction(() => !document.querySelector('.tr-run'), null, {timeout: 120000});
    const card = await (await fetch(base + '/api/training/models/mine:copper-quick')).json();
    assert.equal(card.card.notes, 'one epoch from the browser test');
    assert.equal(card.card.hparams.crop_size, 64);
    assert.equal(card.card.parent, 'mine:copper-ft');

    // ----- The picker, as the Workspace opens it.
    const chosen = page.evaluate(() => import('/static/model_picker.js').then(m => m.openModelPicker({toast() {}}, {setup: 'lab:rig', role: 'body', current: 'mine:copper-ft'})));
    const picker = page.locator('dialog.model-picker');
    await picker.locator('tr[data-ref="lab:body"]').waitFor();
    assert.match(await picker.locator('tr[data-ref="mine:copper-ft"]').innerText(), /in use/);
    assert(await picker.locator('tr[data-ref="lab:seg"] button').isDisabled());  // no body outputs
    await picker.locator('.mp-benchmark').selectOption('mine:copper-b1');
    await shot(page, 'picker');
    await picker.locator('tr[data-ref="lab:body"]').getByRole('button', {name: 'Use'}).click();
    assert.equal(await chosen, 'lab:body');
    assert.equal(await page.locator('dialog.model-picker').count(), 0);
    const cancelled = page.evaluate(() => import('/static/model_picker.js').then(m => m.openModelPicker({toast() {}}, {setup: 'lab:rig', role: 'mask'})));
    await page.locator('dialog.model-picker').getByRole('button', {name: 'Cancel'}).click();
    assert.equal(await cancelled, null);

    assert.deepEqual(errors, []);
    console.log('training page: ok');
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
    fs.rmSync(root, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exit(1); });
