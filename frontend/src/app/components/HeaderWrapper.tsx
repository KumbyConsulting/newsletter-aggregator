'use client';

import { Suspense } from 'react';
import Header from './Header';

/**
 * HeaderWrapper - Wraps the Header component in a Suspense boundary
 * This allows the header to be rendered client-side without blocking the page
 */
export default function HeaderWrapper() {
  return (
    <Suspense fallback={<div className="header-placeholder" style={{ height: 64 }}></div>}>
      <Header />
    </Suspense>
  );
} 