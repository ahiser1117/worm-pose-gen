// Run against the synthetic phase4_fixture.py app; writes inspection screenshots to /tmp.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});
  try {
    const page=await browser.newPage();
    await page.goto(process.env.UI_BASE_URL||'http://127.0.0.1:18776');
    await page.waitForFunction(()=>state.runName==='demo'&&state.frame);
    await page.evaluate(()=>showTab('rerun'));
    await page.locator('#region-algorithm').selectOption('tracked_head');
    assert(await page.locator('#region-algorithm-help').isHidden());
    assert.match(await page.locator('#region-algorithm').getAttribute('title'), /Fits forward/);
    for(const [label,width,height] of [['desktop',1440,1000],['narrow',760,900]]) {
      await page.setViewportSize({width,height});
      const panel=page.locator('#region-params');
      await panel.scrollIntoViewIfNeeded();
      const layout=await panel.evaluate(node=>({
        width:node.clientWidth,scrollWidth:node.scrollWidth,
        labels:[...node.querySelectorAll('.param-name')].map(n=>({name:n.textContent,height:n.clientHeight,scrollHeight:n.scrollHeight,wrap:getComputedStyle(n).whiteSpace})),
        help:node.querySelectorAll('[aria-describedby]').length,
        visibleHelp:[...node.querySelectorAll('[id^="region-param-help-"]')].filter(n=>n.getClientRects().length).length,
      }));
      assert.equal(layout.scrollWidth,layout.width,`${label}: no horizontal parameter overflow`);
      assert.equal(layout.labels.length,8);
      assert.equal(layout.help,8);
      assert.equal(layout.visibleHelp,0);
      assert.equal(layout.labels[0].name,'Hole filling');
      for(const item of layout.labels){assert.equal(item.wrap,'normal');assert.equal(item.height,item.scrollHeight);}
      await page.screenshot({path:`/tmp/region-panel-${label}.png`});
    }
    console.log('Actual region panel: desktop and narrow wrapping, hover-only help and hole-filling order passed.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
