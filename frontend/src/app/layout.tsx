import { Metadata } from 'next';
import ClientLayout from './layout.client';
import { metadata as siteMetadata } from './metadata';

export const metadata: Metadata = siteMetadata;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <ClientLayout>{children}</ClientLayout>;
} 