import { test, expect } from '@playwright/test';

test.describe('Auth Pages Layout - No Scroll Required', () => {
  test('login page should fit without scrolling', async ({ page }) => {
    await page.goto('/login');

    // Wait for page to load
    await expect(page.locator('h1')).toContainText('Welcome Back');

    // Get viewport and page dimensions
    const viewportSize = page.viewportSize();
    const pageHeight = await page.evaluate(() => document.body.scrollHeight);
    const clientHeight = await page.evaluate(() => document.documentElement.clientHeight);

    console.log('Login Page - Viewport height:', viewportSize?.height);
    console.log('Login Page - Content height:', pageHeight);
    console.log('Login Page - Client height:', clientHeight);

    // Check if content requires scrolling
    const requiresScroll = pageHeight > clientHeight;

    if (requiresScroll) {
      // Find which elements are causing overflow
      const footer = await page.locator('footer').boundingBox();
      const lastElement = await page.locator('footer').evaluate(el => {
        return {
          bottom: el.getBoundingClientRect().bottom,
          viewportHeight: window.innerHeight
        };
      });

      console.log('Footer position:', footer);
      console.log('Last element bottom vs viewport:', lastElement);
    }

    expect(requiresScroll).toBe(false);
  });

  test('signup page should fit without scrolling', async ({ page }) => {
    await page.goto('/signup');

    // Wait for page to load
    await expect(page.locator('h1')).toContainText('Get Started Free');

    // Get viewport and page dimensions
    const viewportSize = page.viewportSize();
    const pageHeight = await page.evaluate(() => document.body.scrollHeight);
    const clientHeight = await page.evaluate(() => document.documentElement.clientHeight);

    console.log('Signup Page - Viewport height:', viewportSize?.height);
    console.log('Signup Page - Content height:', pageHeight);
    console.log('Signup Page - Client height:', clientHeight);

    // Check if content requires scrolling
    const requiresScroll = pageHeight > clientHeight;

    if (requiresScroll) {
      // Find which elements are causing overflow
      const footer = await page.locator('footer').boundingBox();
      const lastElement = await page.locator('footer').evaluate(el => {
        return {
          bottom: el.getBoundingClientRect().bottom,
          viewportHeight: window.innerHeight
        };
      });

      console.log('Footer position:', footer);
      console.log('Last element bottom vs viewport:', lastElement);
    }

    expect(requiresScroll).toBe(false);
  });
});
