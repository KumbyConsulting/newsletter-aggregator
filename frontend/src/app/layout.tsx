import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { ConfigProvider, Layout, Grid } from 'antd';
import "./globals.css";

// Initialize the Inter font with extended options
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-inter",
});

// Initialize responsive breakpoints from Ant Design
const { useBreakpoint } = Grid;

export const metadata: Metadata = {
  title: "Kumby Consulting Newsboard | Newsletter Aggregator",
  description: "A modern newsletter aggregator and knowledge base that collects and organizes industry news from various sources",
  keywords: "consulting, newsletter, aggregator, news, articles, rss, feeds, topics, sources, business, industry, research",
  authors: [{ name: "Kumby Consulting Team" }],
  openGraph: {
    type: "website",
    title: "Kumby Consulting Newsboard | Newsletter Aggregator",
    description: "A modern newsletter aggregator and knowledge base that collects and organizes industry news from various sources",
    siteName: "Kumby Consulting Newsboard",
  },
  twitter: {
    card: "summary_large_image",
    title: "Kumby Consulting Newsboard | Newsletter Aggregator",
    description: "A modern newsletter aggregator and knowledge base that collects and organizes industry news from various sources",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#2b7de9",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable}`}>
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <meta name="theme-color" content="#2b7de9" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
      </head>
      <body className={`${inter.className} antialiased`} style={{ margin: 0, padding: 0 }}>
        <ConfigProvider
          theme={{
            token: {
              colorPrimary: '#2b7de9',
              borderRadius: 6,
              fontFamily: inter.style.fontFamily,
            },
            components: {
              Layout: {
                bodyBg: '#f5f5f5',
                headerBg: '#fff',
                footerBg: '#f5f5f5',
                headerPadding: '0 50px',
                headerHeight: 64,
              },
            },
          }}
        >
          <Layout style={{ minHeight: '100vh' }}>
            {children}
          </Layout>
        </ConfigProvider>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Check for dark mode preference
              if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                document.documentElement.classList.add('dark')
              } else {
                document.documentElement.classList.remove('dark')
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
