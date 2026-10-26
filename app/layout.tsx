import type { Metadata, Viewport } from "next";
// Self-hosted so the page has no third-party font request and renders the
// same whether or not Google Fonts is reachable.
import "@fontsource/anton/400.css";
import "@fontsource/archivo/400.css";
import "@fontsource/archivo/600.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Which Bollywood star is your face?",
  description:
    "Upload a photo and find your closest Bollywood lookalike. Face detection and matching run entirely in your browser — the photo never leaves your device.",
};

export const viewport: Viewport = {
  themeColor: "#14123a",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
