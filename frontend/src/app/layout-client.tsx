'use client';

import { Inter } from "next/font/google";
import { ConfigProvider, Layout } from 'antd';
import "./globals.css";
import dynamic from 'next/dynamic';
import DOMPurify from 'isomorphic-dompurify';
import { ThemeProvider } from 'next-themes';

const Header = dynamic(() => import('./components/Header'), { ssr: false });
const { Content, Footer } = Layout;

// Initialize the Inter font with extended options
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-inter",
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable}`} suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <meta name="theme-color" content="#2b7de9" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
      </head>
      <body className={`${inter.className} antialiased`} style={{ margin: 0, padding: 0 }}>
        <ThemeProvider attribute="class">
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
              <Header />
              <Content style={{ padding: '24px', minHeight: 'calc(100vh - 64px - 70px)' }}>
                {children}
              </Content>
              <Footer style={{ textAlign: 'center', padding: '24px 50px' }}>
                Kumby Consulting Newsboard ©{new Date().getFullYear()} - Newsletter Aggregator
              </Footer>
            </Layout>
          </ConfigProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
