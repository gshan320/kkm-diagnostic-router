import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KKM Diagnostic Router & CPG Comparative Intelligence",
  description:
    "Local RAG over Malaysian MOH Clinical Practice Guidelines, Paediatric Protocols, MTS 2022 and the FUKKM formulary.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
