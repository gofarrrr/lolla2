const { chromium } = require('playwright');

(async () => {
  const traceId = process.env.TRACE_ID;
  if (!traceId) {
    console.error('TRACE_ID env var required');
    process.exit(1);
  }
  const url = `http://localhost:3001/analysis/${traceId}/dashboard`;
  const outPath = process.env.OUT_PATH || `evidence_${traceId}.png`;

  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();

  try {
    await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });

    // Click the Evidence tab (use the TabBar within the main region)
    await page.getByRole('main').getByRole('button', { name: 'Evidence' }).click();

    // Wait for the Evidence Ledger heading
    await page.getByRole('heading', { name: 'Evidence Ledger' }).waitFor({ timeout: 60000 });

    // Optionally wait for at least one evidence list item to render
    const items = page.locator('section:has(h3:has-text("Evidence Ledger")) li');
    await items.first().waitFor({ timeout: 60000 });

    // Screenshot the Evidence section only
    const section = page.locator('section:has(h3:has-text("Evidence Ledger"))');
    await section.screenshot({ path: outPath });

    console.log(`Saved screenshot to ${outPath}`);
  } catch (e) {
    console.error('Screenshot failed:', e);
    await page.screenshot({ path: outPath.replace('.png', '_full.png'), fullPage: true }).catch(() => {});
    process.exit(2);
  } finally {
    await browser.close();
  }
})();
