// Run against tests/browser/phase4_fixture.py, never a production workspace.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}}), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.UI_BASE_URL || 'http://127.0.0.1:18770');
    await page.waitForFunction(() => state.runName === 'demo' && state.frame);
    await page.evaluate(() => { showTab('inspect'); setRegionRows(2, 3, 'browser check'); });
    await page.locator('#frame-index').fill('2');
    await page.locator('#go').click();
    await page.waitForFunction(() => state.row === 2);
    assert.match(await page.locator('#context-frame').innerText(), /frame 2/);
    assert.match(await page.locator('#selection-info').innerText(), /Selected 2–3/);
    assert.equal(await page.getByRole('spinbutton', {name: 'Low IoU threshold'}).count(), 1);
    await page.locator('#frame-index').fill('999');
    await page.locator('#go').click();
    assert.equal(await page.locator('#frame-index').getAttribute('aria-invalid'), 'true');
    assert.match(await page.locator('#frame-error').innerText(), /Enter a frame from 0 to 5/);
    assert.match(await page.locator('#frame-index').getAttribute('aria-describedby'), /frame-error/);
    await page.locator('#tabs [data-tab="paint"]').click();
    assert(await page.locator('#frame-error').isHidden());
    await page.waitForFunction(() => !document.querySelector('#mask-clear-draft').disabled);
    assert.equal(await page.getByRole('slider', {name: /Diameter/}).count(), 1);
    assert.equal(await page.getByRole('slider', {name: /Network threshold/}).count(), 1);
    await page.locator('#mask-clear-draft').click();
    assert.match(await page.locator('#context-save').innerText(), /Unsaved mask draft/);
    await page.locator('#tabs [data-tab="review"]').click();
    assert.match(await page.locator('#context-save').innerText(), /Unsaved mask draft/);
    assert.equal(await page.locator('#selection-first').inputValue(), '2');
    assert.equal(await page.locator('#selection-last').inputValue(), '3');
    await page.locator('#tabs [data-tab="rerun"]').click();
    await page.waitForFunction(() => !document.querySelector('#workflow-model-use').disabled);
    await page.locator('#workflow-model-use').click();
    assert.equal(await page.evaluate(() => maskEditor.isDirty()), true, 'Changing a model must not discard an unsaved mask');
    assert.match(await page.locator('#toast').textContent(), /Save or discard the mask draft/);
    await page.locator('#tabs [data-tab="paint"]').click();
    await page.locator('#mask-discard').click();
    await page.evaluate(() => openRightTab('layers'));
    assert.equal(await page.getByRole('slider', {name: /Threshold override/}).count(), 1);
    await page.locator('#app-header [data-screen="training"]').click();
    assert(await page.locator('#details').isHidden());
    await page.locator('#training-jobs').click();
    assert(await page.locator('#right-jobs').isVisible());
    assert(await page.locator('#right-statistics').isHidden());
    await page.locator('#return-workspace').click();
    assert(await page.locator('#right-layers').isVisible());
    assert.equal(await page.evaluate(() => state.rightTab), 'layers');
    await page.evaluate(() => setStatus('Could not save. Try again after the job finishes.', 'error'));
    assert(await page.locator('#toast').isVisible());
    assert.equal(await page.locator('#toast').getAttribute('role'), 'alert');
    assert.equal(await page.locator('#toast').evaluate(n => n.closest('#app-status') !== null), true);
    await page.getByRole('button', {name: 'Dismiss notification'}).click();
    assert(await page.locator('#toast').isHidden());
    await page.evaluate(() => setStatus('Old task error. Retry.', 'error'));
    await page.locator('#app-header [data-screen="import"]').click();
    assert(await page.locator('#toast').isHidden());
    assert(await page.locator('#details').isHidden());
    await page.locator('#app-header [data-screen="open"]').click();
    assert.equal(await page.getByRole('listbox', {name: 'Workspace or read-only run'}).count(), 1);
    const refreshGlyphs = await page.locator('button').evaluateAll(buttons => buttons.filter(b => b.textContent.trim() === '↻').length);
    assert.equal(refreshGlyphs, 0);
    const out = process.env.UI_SCREENSHOTS;
    if (out) fs.mkdirSync(out, {recursive: true});
    for (const task of ['inspect', 'paint', 'rerun', 'review', 'export']) {
      await page.evaluate(task => showTab(task), task);
      if (out) await page.screenshot({path: `${out}/${task}.png`, fullPage: true});
    }
    for (const screen of ['import', 'open', 'labels', 'training']) {
      await page.evaluate(screen => showTab(screen), screen);
      if (out) await page.screenshot({path: `${out}/${screen}.png`, fullPage: true});
    }
    for (const width of [800, 390, 320]) {
      await page.setViewportSize({width, height: 900});
      for (const task of ['inspect', 'paint', 'rerun', 'review', 'export', 'import', 'open', 'labels', 'training']) {
        await page.evaluate(task => showTab(task), task);
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${task} overflow at ${width}`);
      }
    }
    assert.deepEqual(errors, []);
    console.log('PASS shared accessible names, context/draft/range retention, local frame errors, dismissible task errors, Training Jobs/inspector restoration, nine-screen narrow layouts.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
