'use client';

import { Pagination } from 'antd';
import type { PaginationProps } from 'antd';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { useCallback, useEffect } from 'react';

interface PaginationControlsProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  itemsPerPage: number;
  disabled?: boolean;
}

export default function PaginationControls({
  currentPage,
  totalPages,
  totalItems,
  itemsPerPage,
  disabled = false
}: PaginationControlsProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Log props on mount and when they change
  useEffect(() => {
    console.log('PaginationControls props:', { 
      currentPage, 
      totalPages, 
      totalItems, 
      itemsPerPage, 
      disabled 
    });
  }, [currentPage, totalPages, totalItems, itemsPerPage, disabled]);

  const handlePageChange = useCallback((page: number, pageSize?: number) => {
    try {
      console.log(`Page change requested: ${page}, pageSize: ${pageSize || itemsPerPage}`);
      
      // Validate page number
      if (page < 1 || page > totalPages) {
        console.error(`Invalid page number: ${page}`);
        return;
      }
      
      // Create a new URLSearchParams object from the current search params
      const current = new URLSearchParams(Array.from(searchParams.entries()));
      
      // Update the page parameter
      current.set('page', page.toString());
      
      // Optionally update limit if page size changes
      if (pageSize && pageSize !== itemsPerPage) {
        current.set('limit', pageSize.toString());
        // Reset to page 1 when changing page size
        current.set('page', '1');
      }
      
      // Create the new URL with updated parameters
      const query = current.toString();
      const newUrl = `${pathname}?${query}`;
      
      console.log(`Navigating to: ${newUrl}`);
      router.push(newUrl);
    } catch (error) {
      console.error('Error during pagination:', error);
    }
  }, [searchParams, router, pathname, itemsPerPage, totalPages]);

  // Always render pagination controls, even with one page
  // This ensures the UI is consistent and users can still change page size
  return (
    <div className="pagination-container">
      <Pagination
        current={currentPage}
        total={totalItems}
        pageSize={itemsPerPage}
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
        @media (max-width: 576px) {
          .ant-pagination-options {
            margin-left: 8px;
          }
        }
      `}</style>
    </div>
  );
} 