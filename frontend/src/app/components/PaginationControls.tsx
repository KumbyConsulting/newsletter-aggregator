'use client';

import { Pagination } from 'antd';
import type { PaginationProps } from 'antd';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { useCallback } from 'react';

interface PaginationControlsProps {
  current: number;
  total: number;
  pageSize: number;
  onChange: (page: number, pageSize?: number) => void;
  disabled?: boolean;
}

export default function PaginationControls({
  current,
  total,
  pageSize,
  onChange,
  disabled = false
}: PaginationControlsProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const handlePageChange = useCallback((page: number, size?: number) => {
    // Create a new URLSearchParams object from the current search params
    const params = new URLSearchParams(Array.from(searchParams.entries()));
    
    // Update the page parameter
    params.set('page', page.toString());
    
    // Update limit if page size changes
    if (size && size !== pageSize) {
      params.set('limit', size.toString());
    }
    
    // Create the new URL with updated parameters
    const query = params.toString();
    const newUrl = `${pathname}?${query}`;
    
    // Update URL without full page reload
    router.push(newUrl, { scroll: false });
    
    // Call the onChange handler
    onChange(page, size);
  }, [searchParams, router, pathname, pageSize, onChange]);

  return (
    <div className="pagination-container">
      <Pagination
        current={current}
        total={total}
        pageSize={pageSize}
        onChange={handlePageChange}
        onShowSizeChange={(current, size) => handlePageChange(1, size)}
        showSizeChanger={true}
        pageSizeOptions={['10', '20', '30', '50']}
        responsive
        showTotal={(total, range) => `${range[0]}-${range[1]} of ${total} items`}
        disabled={disabled}
        style={{
          textAlign: 'center',
          marginTop: '24px',
          marginBottom: '24px'
        }}
      />
      <style jsx global>{`
        .pagination-container {
          width: 100%;
          display: flex;
          justify-content: center;
          margin: 24px 0;
        }
        .ant-pagination {
          display: inline-flex;
          align-items: center;
        }
        .ant-pagination-options {
          margin-left: 16px;
        }
      `}</style>
    </div>
  );
} 