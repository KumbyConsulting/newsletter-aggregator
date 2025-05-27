'use client';

import { ConfigProvider, Layout } from 'antd';
import "./globals.css";
import dynamic from 'next/dynamic';
import DOMPurify from 'isomorphic-dompurify';
import { ThemeProvider } from 'next-themes';
import { theme } from './theme';

const Header = dynamic(() => import('./components/Header'), { ssr: false });
const { Content, Footer } = Layout;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <meta name="theme-color" content="#00405e" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700;900&family=DM+Serif+Display:ital,wght@0,400;1,400&display=swap" rel="stylesheet" />
      </head>
      <body className="antialiased" style={{ margin: 0, padding: 0, fontFamily: 'DM Sans, Arial, Helvetica, sans-serif' }}>
        <ThemeProvider attribute="class">
          <ConfigProvider theme={theme}>
            <Layout style={{ minHeight: '100vh' }}>
              <Header />
              <Content style={{ minHeight: 'calc(100vh - 64px - 70px)', background: 'none' }}>
                <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 1rem', width: '100%' }}>
                  {children}
                </div>
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
