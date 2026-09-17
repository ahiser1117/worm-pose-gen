// Real layer compositor and Paint controller; all image data is synthetic.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setContent('<canvas id="canvas" width="64" height="64" style="width:64px;height:64px"></canvas><div id="caption"></div><div id="legend"></div><div id="layers"></div><label><input id="mask-enable" type="checkbox" checked>Paint</label><label><input id="mask-visible" type="checkbox" checked>Editable mask</label><select id="mask-opacity"><option value="45">45%</option><option value="20">20%</option><option value="0">Off</option></select>');
    for (const file of ['api.js', 'layers.js', 'masks.js']) {
      await page.addScriptTag({content: fs.readFileSync(`src/worm_pose_gen/pose_viewer_ui/${file}`, 'utf8')});
    }
    await page.evaluate(async () => {
      state.info = {server: 'app'}; state.sourceKind = 'workspace'; state.runName = 'demo';
      state.run = {entry: {recording: 'demo'}, series: {frame_index: [0]}};
      state.frame = {row: 0, frame_index: 0, detail: 'full', stats: {}};
      state.decoded = {width: 64, height: 64, mask_raw: new Uint8Array(64 * 64)};
      const image = document.createElement('canvas'); image.width = image.height = 64;
      const g = image.getContext('2d'); g.fillStyle = 'black'; g.fillRect(0, 0, 64, 64);
      state.image = await loadImage(image.toDataURL());
      g.fillStyle = 'white'; g.fillRect(10, 10, 10, 10);
      for (let y = 10; y < 20; y++) for (let x = 10; x < 20; x++) state.decoded.mask_raw[y * 64 + x] = 255;
      const label = image.toDataURL();
      window.currentFrameIndex = () => 0;
      window.serverIsApp = () => true;
      window.isWorkspace = () => true;
      post = async () => ({target: {workspace: 'demo', frame: 0}, frame: 0, width: 64, height: 64, image: label, mask: label});
      for (const l of LAYERS) l.on = ['image', 'editable_mask'].includes(l.id);
      renderLayers(); maskEditor.init(); await maskEditor.onFrame(true); buildOverlay(); draw();
      window.pixel = (x = 15, y = 15) => Array.from(ctx.getImageData(x, y, 1, 1).data);
    });
    const editable = page.getByRole('checkbox', {name: 'Editable mask (Paint)', exact: true});
    const opacity = page.getByRole('slider', {name: 'Editable mask (Paint) opacity', exact: true});
    assert(await editable.isChecked());
    assert.match(await page.locator('#legend').innerText(), /Editable mask/);
    const magenta = await page.evaluate(() => pixel());
    assert(magenta[0] > magenta[1] * 2 && magenta[2] > magenta[1]);

    await editable.uncheck();
    assert.equal(await page.locator('#mask-visible').isChecked(), false);
    assert.deepEqual(await page.evaluate(() => pixel()), [0, 0, 0, 255]);
    await page.locator('#mask-visible').check();
    assert(await editable.isChecked());
    assert.deepEqual(await page.evaluate(() => pixel()), magenta);

    await opacity.evaluate(node => { node.value = '.75'; node.dispatchEvent(new Event('input')); });
    assert.equal(await page.locator('#mask-opacity').inputValue(), '75');
    assert((await page.evaluate(() => pixel()))[0] > magenta[0]);
    await page.locator('#mask-opacity').selectOption('20');
    assert.equal(await opacity.inputValue(), '0.2');
    assert.equal(await page.evaluate(() => maskEditor.cycleOpacity()), 0);
    assert.equal(await opacity.inputValue(), '0');
    assert.equal(await page.locator('#mask-opacity').inputValue(), '0');
    assert(!/Editable mask/.test(await page.locator('#legend').innerText()));
    await page.locator('#mask-opacity').selectOption('45');

    // The blue segmentation mask and the editable labels have separate controls.
    await editable.uncheck();
    await page.getByRole('checkbox', {name: 'Raw mask (threshold)', exact: true}).check();
    const blue = await page.evaluate(() => pixel());
    assert(blue[2] > blue[0] * 2);
    await page.getByRole('checkbox', {name: 'Raw mask (threshold)', exact: true}).uncheck();
    await editable.check();

    // Leaving Paint must hide the mask without altering an unsaved stroke.
    const canvas = await page.locator('#canvas').boundingBox();
    await page.mouse.click(canvas.x + 30, canvas.y + 30);
    assert.equal(await page.evaluate(() => maskEditor.isDirty()), true);
    const draftPixel = await page.evaluate(() => pixel(30, 30));
    assert(draftPixel[0] > 0);
    await page.evaluate(() => maskEditor.setActive(false));
    assert.deepEqual(await page.evaluate(() => pixel(30, 30)), [0, 0, 0, 255]);
    assert.equal(await page.evaluate(() => maskEditor.isDirty()), true);
    assert(!/Editable mask/.test(await page.locator('#legend').innerText()));
    assert(await page.locator('.layer[data-layer="editable_mask"]').evaluate(node => node.classList.contains('unavailable')));
    await page.evaluate(() => maskEditor.setActive(true));
    await page.mouse.move(canvas.x + 60, canvas.y + 60); // move the brush cursor away
    assert.deepEqual(await page.evaluate(() => pixel(30, 30)), draftPixel);
    assert(await editable.isChecked());

    // Motion suppression still applies to Paint's separate canvas overlay.
    await page.evaluate(() => { state.playing = 1; draw(); });
    assert.deepEqual(await page.evaluate(() => pixel(15, 15)), [0, 0, 0, 255]);
    await page.evaluate(() => { state.playing = null; state.timeline.dragging = true; draw(); });
    assert.deepEqual(await page.evaluate(() => pixel(15, 15)), [0, 0, 0, 255]);
    await page.evaluate(() => { state.timeline.dragging = false; draw(); });
    assert.deepEqual(await page.evaluate(() => pixel(15, 15)), magenta);

    // Restoring a saved view uses the same source of truth as both control sets.
    await page.evaluate(() => { Object.assign(layer('editable_mask'), {on: false, alpha: .65}); renderLayers(); draw(); });
    assert.equal(await page.locator('#mask-visible').isChecked(), false);
    assert.equal(await page.locator('#mask-opacity').inputValue(), '65');
    assert.equal(await opacity.inputValue(), '0.65');
    assert.deepEqual(errors, []);
    console.log('Editor layers: independent blue/magenta masks, synchronized visibility/opacity, Paint-only rendering, retained drafts, motion suppression and view restoration passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
