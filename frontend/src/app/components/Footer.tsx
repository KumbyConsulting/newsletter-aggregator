'use client';

import { Layout } from 'antd';
const { Footer: AntFooter } = Layout;

export default function Footer() {
  return (
    <AntFooter style={{ textAlign: 'center', padding: '24px 50px' }}>
      Kumby Consulting Newsboard ©{new Date().getFullYear()} - Newsletter Aggregator
    </AntFooter>
  );
} 