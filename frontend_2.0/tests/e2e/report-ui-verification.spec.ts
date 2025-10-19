import { test, expect } from '@playwright/test';

/**
 * OPERATION RADIANCE - Phase 2B: UI Rendering Verification
 *
 * This test navigates to a completed report and verifies all UI sections:
 * - Clicks all navigation buttons
 * - Takes screenshots at each step
 * - Documents which sections display data vs. which are empty
 *
 * Target: trace_id 7c1cfa21-857f-40e8-b3b5-4827da0a1ff5
 */

test.describe('Report UI Verification - Operation Radiance Phase 2B', () => {
  const TRACE_ID = '7c1cfa21-857f-40e8-b3b5-4827da0a1ff5';
  const REPORT_URL = `/analysis/${TRACE_ID}/report_v2`;

  test('Navigate to report and capture all UI sections', async ({ page }) => {
    console.log('🚀 Starting UI verification test');

    // Navigate to the report
    await page.goto(REPORT_URL);
    console.log(`✅ Navigated to: ${REPORT_URL}`);

    // Wait for the page to load fully
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // Additional wait for any dynamic content

    // Take initial screenshot
    await page.screenshot({
      path: 'ui-verification/01-initial-page-load.png',
      fullPage: true
    });
    console.log('📸 Captured: 01-initial-page-load.png');

    let screenshotIndex = 2;

    // Click "See Full Analysis" button to reveal all sections
    try {
      const fullAnalysisBtn = page.getByText('See Full Analysis');
      if (await fullAnalysisBtn.isVisible()) {
        console.log('🖱️  Clicking: "See Full Analysis" button');
        await fullAnalysisBtn.click();
        await page.waitForTimeout(1000);

        const screenshotNum = String(screenshotIndex).padStart(2, '0');
        await page.screenshot({
          path: `ui-verification/${screenshotNum}-see-full-analysis.png`,
          fullPage: true
        });
        console.log(`📸 Captured: ${screenshotNum}-see-full-analysis.png`);
        screenshotIndex++;
      }
    } catch (e) {
      console.log('⚠️  Could not click "See Full Analysis" button:', e);
    }

    // Navigation sections to click through
    const navigationSections = [
      'Executive Summary',
      'Recommendations',
      'Key Assumptions',
      'Consultant Analyses',
      'Critical Analysis',
      'Enhancement Research',
      'Human Input Impact',
      'Process Transparency',
      'Quality Metrics'
    ];

    console.log(`\n🔍 Found ${navigationSections.length} navigation sections to test`);

    // Click each navigation button in the sidebar
    for (const sectionName of navigationSections) {
      try {
        // Find button by text in the sidebar (col-span-3)
        const navButton = page.locator('.col-span-3 button').filter({ hasText: sectionName });

        const isVisible = await navButton.isVisible().catch(() => false);

        if (!isVisible) {
          console.log(`⏭️  Skipping "${sectionName}" (not visible)`);
          continue;
        }

        console.log(`🖱️  Clicking navigation: "${sectionName}"`);
        await navButton.click();

        // Wait for content to render
        await page.waitForTimeout(1000);

        // Take screenshot
        const screenshotNum = String(screenshotIndex).padStart(2, '0');
        const filename = `ui-verification/${screenshotNum}-section-${sectionName.toLowerCase().replace(/\s+/g, '-')}.png`;
        await page.screenshot({
          path: filename,
          fullPage: true
        });
        console.log(`📸 Captured: ${filename}`);

        screenshotIndex++;
      } catch (e) {
        console.log(`❌ Error clicking "${sectionName}": ${e}`);
      }
    }

    // Try clicking "Enter Glass Box" button
    try {
      const glassBoxBtn = page.getByText('Enter Glass Box');
      if (await glassBoxBtn.isVisible()) {
        console.log('🖱️  Clicking: "Enter Glass Box" button');
        await glassBoxBtn.click();
        await page.waitForTimeout(2000); // Wait for forensics data to load

        const screenshotNum = String(screenshotIndex).padStart(2, '0');
        await page.screenshot({
          path: `ui-verification/${screenshotNum}-glass-box-view.png`,
          fullPage: true
        });
        console.log(`📸 Captured: ${screenshotNum}-glass-box-view.png`);
        screenshotIndex++;
      }
    } catch (e) {
      console.log('⚠️  Could not click "Enter Glass Box" button:', e);
    }

    // Check data presence in the bundle
    console.log('\n🔍 Analyzing data structure from bundle:');

    // Check what data is actually present in the page
    const hasRecommendations = await page.locator('text=/Strategic Recommendations/i').isVisible();
    const hasExecutiveSummary = await page.locator('text=/Executive Summary/i').isVisible();
    const hasConsultants = await page.locator('text=/Consultant Analyses/i').isVisible();

    console.log(`  - Recommendations section visible: ${hasRecommendations}`);
    console.log(`  - Executive Summary visible: ${hasExecutiveSummary}`);
    console.log(`  - Consultants section visible: ${hasConsultants}`);

    // Take final screenshot
    await page.screenshot({
      path: `ui-verification/${String(screenshotIndex).padStart(2, '0')}-final-state.png`,
      fullPage: true
    });
    console.log(`📸 Captured: ${String(screenshotIndex).padStart(2, '0')}-final-state.png`);

    console.log('\n✅ UI verification test complete');
  });

  test('Verify report data against bundle structure', async ({ page }) => {
    console.log('🚀 Starting bundle data verification');

    // Navigate to the report
    await page.goto(REPORT_URL);
    await page.waitForLoadState('networkidle');

    // Check what data is actually rendered
    console.log('\n🔍 Extracting rendered data from UI:');

    // Executive Summary
    try {
      const execSummary = await page.locator('[data-testid="executive-summary"], .executive-summary').textContent();
      console.log(`  Executive Summary: ${execSummary ? 'PRESENT' : 'EMPTY'} (${execSummary?.length || 0} chars)`);
    } catch (e) {
      console.log('  Executive Summary: NOT FOUND');
    }

    // Recommendations count
    try {
      const recommendations = await page.locator('[data-testid="recommendation"], .recommendation').count();
      console.log(`  Recommendations: ${recommendations} items found`);
    } catch (e) {
      console.log('  Recommendations: ERROR');
    }

    // Key Decisions count
    try {
      const decisions = await page.locator('[data-testid="decision"], .decision').count();
      console.log(`  Key Decisions: ${decisions} items found`);
    } catch (e) {
      console.log('  Key Decisions: ERROR');
    }

    // Evidence Trail count
    try {
      const evidence = await page.locator('[data-testid="evidence"], .evidence').count();
      console.log(`  Evidence Trail: ${evidence} items found`);
    } catch (e) {
      console.log('  Evidence Trail: ERROR');
    }

    // Consultant Analyses count
    try {
      const consultants = await page.locator('[data-testid="consultant"], .consultant').count();
      console.log(`  Consultant Analyses: ${consultants} items found`);
    } catch (e) {
      console.log('  Consultant Analyses: ERROR');
    }

    // Devils Advocate
    try {
      const devils = await page.locator('[data-testid="devils-advocate"], .devils-advocate').textContent();
      console.log(`  Devils Advocate: ${devils ? 'PRESENT' : 'EMPTY'} (${devils?.length || 0} chars)`);
    } catch (e) {
      console.log('  Devils Advocate: NOT FOUND');
    }

    console.log('\n✅ Bundle data verification complete');
  });
});
