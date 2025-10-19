import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

test.describe('Operation Polish - Comprehensive UX Audit', () => {
  const REPORT_URL = 'http://localhost:3001/analysis/05e4839c-9c36-4c06-8017-12b8bfcf430c/report_v2';
  const OUTPUT_DIR = 'test-results/polish-ux-audit';

  test.beforeAll(() => {
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }
  });

  test('Comprehensive UX Audit with P0 Verification', async ({ page }) => {
    console.log('🚀 Starting comprehensive UX audit');

    // Set viewport to standard desktop
    await page.setViewportSize({ width: 1920, height: 1080 });

    // Navigate to report
    await page.goto(REPORT_URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000); // Wait for any client-side rendering

    console.log('✅ Navigated to report page');

    // CAPTURE 1: Initial page load (full viewport)
    await page.screenshot({
      path: path.join(OUTPUT_DIR, '01-initial-full-page.png'),
      fullPage: true
    });
    console.log('📸 Captured: 01-initial-full-page.png');

    // CAPTURE 2: Hero section (query display issue)
    const heroSection = page.locator('body').first();
    await page.screenshot({
      path: path.join(OUTPUT_DIR, '02-hero-query-section.png'),
      clip: { x: 0, y: 0, width: 1920, height: 400 }
    });
    console.log('📸 Captured: 02-hero-query-section.png');

    // EXTRACT: Query text to measure size
    const queryText = await page.locator('h1, .text-3xl, .text-4xl').first().textContent();
    console.log(`📏 Query text length: ${queryText?.length} characters`);
    console.log(`📝 Query text: ${queryText?.substring(0, 100)}...`);

    // CAPTURE 3: Navigation sections (check if visible without scrolling)
    const navSections = [
      'Executive Summary',
      'Recommendations',
      'Key Assumptions',
      'Consultant Analyses',
      'Critical Analysis',
      'Process Transparency'
    ];

    console.log('\n🔍 Checking navigation visibility:');
    for (const section of navSections) {
      const link = page.locator(`a:has-text("${section}")`);
      const isVisible = await link.isVisible();
      const box = isVisible ? await link.boundingBox() : null;
      console.log(`  ${section}: ${isVisible ? '✅ Visible' : '❌ Hidden'} ${box ? `(y: ${Math.round(box.y)})` : ''}`);
    }

    // P0-1 VERIFICATION: Executive Summary markdown rendering
    console.log('\n🔍 P0-1: Verifying Executive Summary markdown rendering');

    // Click Executive Summary in nav
    await page.click('a:has-text("Executive Summary")');
    await page.waitForTimeout(1000);

    // Capture Executive Summary section
    await page.screenshot({
      path: path.join(OUTPUT_DIR, '03-executive-summary-p01.png'),
      fullPage: false
    });
    console.log('📸 Captured: 03-executive-summary-p01.png');

    // Check for raw markdown syntax (BAD)
    const executiveSummaryText = await page.locator('.prose, .border-l-4').first().textContent();
    const hasRawMarkdown = executiveSummaryText?.includes('##') || executiveSummaryText?.includes('\\n\\n');
    console.log(`  Raw markdown detected: ${hasRawMarkdown ? '❌ FAIL' : '✅ PASS'}`);

    // Check for formatted elements (GOOD)
    const hasFormattedHeadings = await page.locator('.prose h2, .prose h3').count() > 0;
    console.log(`  Formatted headings present: ${hasFormattedHeadings ? '✅ PASS' : '❌ FAIL'}`);

    // P0-3 VERIFICATION: Strategic recommendations
    console.log('\n🔍 P0-3: Verifying Strategic Recommendations');

    await page.click('a:has-text("Recommendations")');
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: path.join(OUTPUT_DIR, '04-recommendations-p03.png'),
      fullPage: false
    });
    console.log('📸 Captured: 04-recommendations-p03.png');

    // Extract recommendation text
    const recommendations = await page.locator('.space-y-6 > div').allTextContents();
    console.log(`  Found ${recommendations.length} recommendations`);

    for (let i = 0; i < Math.min(3, recommendations.length); i++) {
      const rec = recommendations[i];
      const isGeneric = rec.includes('Review') || rec.includes('Implement recommendations following');
      console.log(`  Recommendation ${i+1}: ${isGeneric ? '❌ GENERIC' : '✅ SPECIFIC'}`);
      console.log(`    Preview: ${rec.substring(0, 80)}...`);
    }

    // P0-2 VERIFICATION: Process Transparency warning banner
    console.log('\n🔍 P0-2: Verifying Process Transparency warning banner');

    await page.click('a:has-text("Process Transparency")');
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: path.join(OUTPUT_DIR, '05-process-transparency-p02.png'),
      fullPage: false
    });
    console.log('📸 Captured: 05-process-transparency-p02.png');

    // Check for warning banner
    const warningBanner = page.locator('.border-accent:has-text("Limited Process Data")');
    const bannerVisible = await warningBanner.isVisible();
    console.log(`  Warning banner present: ${bannerVisible ? '✅ PASS' : '❌ FAIL'}`);

    if (bannerVisible) {
      const bannerText = await warningBanner.textContent();
      console.log(`  Banner includes "What's real": ${bannerText?.includes("What's real") ? '✅' : '❌'}`);
      console.log(`  Banner includes "What's placeholder": ${bannerText?.includes("What's placeholder") ? '✅' : '❌'}`);
    }

    // UX ANALYSIS: Information Hierarchy
    console.log('\n📊 UX ANALYSIS: Information Hierarchy');

    // Scroll through all sections and capture
    const sections = [
      'Key Assumptions',
      'Consultant Analyses',
      'Critical Analysis',
      'Enhancement Research',
      'Human Input Impact',
      'Quality Metrics'
    ];

    for (const section of sections) {
      try {
        await page.click(`a:has-text("${section}")`);
        await page.waitForTimeout(800);

        const sectionElement = page.locator(`h2:has-text("${section}")`).first();
        const isVisible = await sectionElement.isVisible();

        if (isVisible) {
          const box = await sectionElement.boundingBox();
          console.log(`  ${section}: y=${Math.round(box?.y || 0)} (${box && box.y < 1080 ? 'above fold' : 'below fold'})`);

          await page.screenshot({
            path: path.join(OUTPUT_DIR, `06-section-${section.toLowerCase().replace(/\s+/g, '-')}.png`),
            fullPage: false
          });
        }
      } catch (e) {
        console.log(`  ${section}: ❌ Not found or not clickable`);
      }
    }

    // UX ANALYSIS: Scroll distance to critical content
    console.log('\n📏 UX ANALYSIS: Scroll Distance to Critical Content');

    await page.goto(REPORT_URL);
    await page.waitForTimeout(2000);

    const criticalSections = ['Executive Summary', 'Recommendations'];

    for (const section of criticalSections) {
      const heading = page.locator(`h2:has-text("${section}")`).first();
      const box = await heading.boundingBox();
      if (box) {
        const scrollsRequired = Math.ceil(box.y / 1080);
        console.log(`  ${section}: ${Math.round(box.y)}px (${scrollsRequired} viewport scroll${scrollsRequired > 1 ? 's' : ''})`);
      }
    }

    // CAPTURE: Final full-page screenshot
    await page.screenshot({
      path: path.join(OUTPUT_DIR, '99-final-full-page.png'),
      fullPage: true
    });
    console.log('\n📸 Captured: 99-final-full-page.png');

    // SUMMARY REPORT
    console.log('\n' + '='.repeat(60));
    console.log('📋 OPERATION POLISH - UX AUDIT SUMMARY');
    console.log('='.repeat(60));
    console.log('\nP0 Verification Results:');
    console.log(`  P0-1 (Markdown Rendering): ${hasRawMarkdown ? '❌ FAIL' : '✅ PASS'}`);
    console.log(`  P0-2 (Warning Banner): ${bannerVisible ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`  P0-3 (Recommendations): Check manual review above`);
    console.log('\nUX Issues Identified:');
    console.log('  - Query text size and scroll distance');
    console.log('  - Information hierarchy and prioritization');
    console.log('  - Navigation visibility and prominence');
    console.log('\nAll screenshots saved to:', OUTPUT_DIR);
    console.log('='.repeat(60));
  });
});
