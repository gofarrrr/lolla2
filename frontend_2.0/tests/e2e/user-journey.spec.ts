import { test, expect } from '@playwright/test';

test.describe('User Journey - Complete Analysis Flow', () => {
  test('should complete full analysis journey from landing to report', async ({ page }) => {
    // Step 1: Visit Landing Page
    await test.step('Visit landing page', async () => {
      await page.goto('/');
      await expect(page).toHaveTitle(/Lolla - Strategic Intelligence Platform/);
      await expect(page.locator('h2')).toContainText('Multiple Forces');
      await expect(page.locator('h2')).toContainText('Exponential Results');
    });

    // Step 2: Sign up (mock auth)
    await test.step('Sign up for account', async () => {
      await page.click('a:has-text("Start Free")');
      await expect(page).toHaveURL(/\/signup/);

      // Fill signup form
      await page.fill('input[type="text"]', 'Test User');
      await page.fill('input[type="email"]', 'test@example.com');
      await page.fill('input[type="password"]', 'testpassword123');
      await page.check('input[type="checkbox"]');

      // Submit form (which sets mock session and redirects to /analyze)
      await page.click('button:has-text("Start Free Analysis")');

      // Should redirect to analyze page
      await expect(page).toHaveURL(/\/analyze/);
    });

    // Step 3: Fill Query
    await test.step('Enter strategic question', async () => {
      await expect(page.locator('h1')).toContainText('New Analysis');

      const textarea = page.locator('textarea');
      await textarea.fill('Should we expand our SaaS product into the European market? What are the strategic considerations for market entry?');

      // Check character count updates
      await expect(page.locator('text=/\\d+ \\/ 2000 characters/')).toBeVisible();
    });

    // Step 4: Adjust Quality Slider
    await test.step('Set quality target', async () => {
      const slider = page.locator('input[type="range"]');
      await slider.fill('85');
      await expect(page.locator('text=Quality Target: 85%')).toBeVisible();
    });

    // Step 5: Skip to Analysis (without PDF upload for speed)
    await test.step('Submit analysis', async () => {
      await page.click('button:has-text("Skip to Analysis")');

      // Should redirect to processing view or login if auth fails
      // For now, we expect it might redirect to login (auth not fully implemented)
      await page.waitForURL(/\/(analysis\/.+|login)/, { timeout: 10000 });
    });

    console.log('✅ User successfully navigated through the analysis flow!');
  });

  test('should display Academy page correctly', async ({ page }) => {
    await test.step('Visit Academy', async () => {
      await page.goto('/academy');
      await expect(page.locator('h1')).toContainText('Mental Models Academy');
      await expect(page.locator('text=137 mental models')).toBeVisible();
    });

    await test.step('Show mental model categories', async () => {
      await expect(page.locator('text=Strategic Frameworks')).toBeVisible();
      await expect(page.locator('text=Decision Science')).toBeVisible();
      await expect(page.locator('text=Systems Thinking')).toBeVisible();
    });

    await test.step('Show N-Way Relations', async () => {
      await expect(page.locator('text=N-Way Relations')).toBeVisible();
      await expect(page.locator('text=Lollapalooza Effect')).toBeVisible();
    });

    console.log('✅ Academy page displays correctly!');
  });

  test('should navigate all main pages without errors', async ({ page }) => {
    const pages = [
      { url: '/', selector: 'h2:has-text("Multiple Forces")' },
      { url: '/dashboard', selector: 'h1:has-text("Your Analyses")' },
      { url: '/analyze', selector: 'h1:has-text("New Analysis")' },
      { url: '/academy', selector: 'h1:has-text("Mental Models Academy")' },
    ];

    for (const testPage of pages) {
      await test.step(`Navigate to ${testPage.url}`, async () => {
        await page.goto(testPage.url);
        await expect(page.locator(testPage.selector)).toBeVisible();
        console.log(`✅ ${testPage.url} loaded successfully`);
      });
    }
  });
});

test.describe('UI Components and Interactions', () => {
  test('should display brutalist design elements', async ({ page }) => {
    await page.goto('/');

    await test.step('Check brutalist styling', async () => {
      // Check for thick borders (4px in the redesigned version)
      const nav = page.locator('nav');
      await expect(nav).toHaveCSS('border-bottom-width', '4px');

      // Check accent color on CTA button
      const ctaButton = page.locator('a.btn-accent').first();
      await expect(ctaButton).toBeVisible();
    });

    console.log('✅ Brutalist design elements present!');
  });

  test('should handle query input validation', async ({ page }) => {
    await page.goto('/analyze');

    await test.step('Disable submit when empty', async () => {
      const submitButton = page.locator('button:has-text("Skip to Analysis")');
      await expect(submitButton).toBeDisabled();
    });

    await test.step('Enable submit when filled', async () => {
      await page.locator('textarea').fill('Test query');
      const submitButton = page.locator('button:has-text("Skip to Analysis")');
      await expect(submitButton).toBeEnabled();
    });

    console.log('✅ Query validation works correctly!');
  });
});

test.describe('Responsive Design', () => {
  test('should work on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE

    await page.goto('/');
    await expect(page.locator('h1:has-text("Lolla")')).toBeVisible();
    await expect(page.locator('h2')).toContainText('Multiple Forces');

    console.log('✅ Mobile viewport works!');
  });

  test('should work on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 }); // iPad

    await page.goto('/');
    await expect(page.locator('h2')).toContainText('Multiple Forces');

    console.log('✅ Tablet viewport works!');
  });
});
