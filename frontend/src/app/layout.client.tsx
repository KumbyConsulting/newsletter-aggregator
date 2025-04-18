'use client';

import { Inter } from "next/font/google";
import { ConfigProvider, Layout, Grid } from 'antd';
import "./globals.css";
import HeaderWrapper from "./components/HeaderWrapper";
import Footer from './components/Footer';
import DOMPurify from 'isomorphic-dompurify';
import { ThemeProvider } from 'next-themes';
import { theme } from './theme';

const { Content } = Layout;

// Initialize the Inter font with extended options
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-inter",
});

// Initialize responsive breakpoints from Ant Design
const { useBreakpoint } = Grid;

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable}`} suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#00405e" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <link rel="icon" href="/favicon.ico" sizes="any" />
      </head>
      <body className={`${inter.className} antialiased`} style={{ margin: 0, padding: 0 }}>
        <ThemeProvider attribute="class">
          <ConfigProvider theme={theme}>
            <Layout style={{ minHeight: '100vh' }}>
              <HeaderWrapper />
              <Content style={{ padding: '24px', minHeight: 'calc(100vh - 64px - 70px)' }}>
                {children}
              </Content>
              <Footer />
            </Layout>
          </ConfigProvider>
        </ThemeProvider>
      </body>
    </html>
  );
} 