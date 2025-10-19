import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({
  subsets: ["latin"],
  variable: '--font-sans',
  display: 'swap',
});

export const metadata: Metadata = {
  title: "Lolla - Strategic Intelligence Platform",
  description: "Multiple Forces. Exponential Results. Be Your Own CEO with AI Consultant Teams.",
  keywords: "strategic analysis, AI consultants, cognitive intelligence, business strategy, Lollapalooza Effect",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased bg-canvas text-ink-1">
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
