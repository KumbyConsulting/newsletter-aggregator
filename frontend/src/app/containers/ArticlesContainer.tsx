'use client';

import { useEffect } from 'react';
import { Pagination, Row, Col, Space, Typography, Select, Input, Button } from 'antd';
import { SearchOutlined, FilterOutlined, SortAscendingOutlined, SortDescendingOutlined } from '@ant-design/icons';
import { ArticlesGrid } from '../components/ArticlesGrid';
import { useArticles } from '../context/ArticleContext';

const { Title } = Typography;
const { Option } = Select;

// Define available sort fields
const SORT_FIELDS = [
  { value: 'pub_date', label: 'Publication Date' },
  { value: 'title', label: 'Title' },
  { value: 'source', label: 'Source' },
  { value: 'topic', label: 'Topic' },
];

// Add a component to inject the styles
const ArticlesContainerStyles = () => {
  useEffect(() => {
    // Only inject the styles once
    if (!document.getElementById('articles-container-styles')) {
      const styleElement = document.createElement('style');
      styleElement.id = 'articles-container-styles';
      styleElement.innerHTML = styles;
      document.head.appendChild(styleElement);
    }
    
    // Clean up on unmount
    return () => {
      const styleElement = document.getElementById('articles-container-styles');
      if (styleElement) {
        styleElement.remove();
      }
    };
  }, []);
  
  return null;
};

export function ArticlesContainer() {
  // Get all article context state and actions
  const {
    articles,
    totalArticles,
    currentPage,
    perPage,
    totalPages,
    isLoading,
    error,
    selectedTopic,
    searchQuery,
    sortBy,
    sortOrder,
    fetchArticles,
    setPage,
    setPerPage,
    setTopic,
    setSearchQuery,
    setSort,
    resetFilters,
  } = useArticles();

  // Fetch articles when dependencies change
  useEffect(() => {
    fetchArticles();
  }, [fetchArticles, currentPage, perPage, selectedTopic, searchQuery, sortBy, sortOrder]);

  // Debounced search handler using useCallback
  const handleSearch = (value: string) => {
    setSearchQuery(value);
  };

  // Toggle sort order
  const toggleSortOrder = () => {
    setSort(sortBy, sortOrder === 'asc' ? 'desc' : 'asc');
  };

  return (
    <div className="articles-container">
      <ArticlesContainerStyles />
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Header with title and controls */}
        <Row gutter={[16, 16]} align="middle" justify="space-between">
          <Col xs={24} md={12}>
            <Title level={2}>Latest News Articles</Title>
          </Col>
          <Col xs={24} md={12}>
            <Row gutter={[8, 8]} justify="end">
              <Col xs={24} sm={18} md={16}>
                <Input
                  placeholder="Search articles..."
                  prefix={<SearchOutlined />}
                  value={searchQuery}
                  onChange={(e) => handleSearch(e.target.value)}
                  allowClear
                />
              </Col>
              <Col xs={12} sm={6} md={8}>
                <Button
                  icon={<FilterOutlined />}
                  onClick={resetFilters}
                  disabled={!searchQuery && selectedTopic === 'All' && sortBy === 'pub_date' && sortOrder === 'desc'}
                >
                  Reset
                </Button>
              </Col>
            </Row>
          </Col>
        </Row>

        {/* Filters row */}
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} md={16}>
            <Space wrap>
              <span>Topic:</span>
              <Select 
                value={selectedTopic} 
                onChange={setTopic} 
                style={{ width: 200 }}
                loading={isLoading}
              >
                <Option value="All">All Topics</Option>
                <Option value="Clinical Trials">Clinical Trials</Option>
                <Option value="Drug Development">Drug Development</Option>
                <Option value="Regulatory">Regulatory</Option>
                <Option value="Manufacturing">Manufacturing</Option>
                <Option value="Market">Market</Option>
                <Option value="R&D">R&D</Option>
                <Option value="Business">Business</Option>
                <Option value="Patents">Patents</Option>
                <Option value="Medical">Medical</Option>
                <Option value="Safety">Safety</Option>
              </Select>
            </Space>
          </Col>
          <Col xs={24} md={8}>
            <Space wrap>
              <span>Sort by:</span>
              <Select 
                value={sortBy} 
                onChange={(value) => setSort(value, sortOrder)}
                style={{ width: 150 }}
              >
                {SORT_FIELDS.map((field) => (
                  <Option key={field.value} value={field.value}>
                    {field.label}
                  </Option>
                ))}
              </Select>
              <Button
                icon={sortOrder === 'asc' ? <SortAscendingOutlined /> : <SortDescendingOutlined />}
                onClick={toggleSortOrder}
              />
            </Space>
          </Col>
        </Row>

        {/* Articles grid */}
        <ArticlesGrid
          articles={articles}
          loading={isLoading}
          error={error || undefined}
          emptyMessage={
            searchQuery
              ? `No articles found matching "${searchQuery}"`
              : selectedTopic !== 'All'
              ? `No articles found for topic "${selectedTopic}"`
              : 'No articles found'
          }
          syncWithBackend={false} // We don't need to sync since we're using the context
        />

        {/* Pagination */}
        {totalArticles > 0 && (
          <Row justify="center">
            <Col>
              <Pagination
                current={currentPage}
                pageSize={perPage}
                total={totalArticles}
                onChange={setPage}
                showSizeChanger
                onShowSizeChange={(_, size) => setPerPage(size)}
                pageSizeOptions={['9', '18', '36', '72']}
                showTotal={(total, range) => `${range[0]}-${range[1]} of ${total} articles`}
              />
            </Col>
          </Row>
        )}
      </Space>
    </div>
  );
}

// Add styles for ArticlesContainer
const styles = `
.articles-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
}

.articles-container .ant-space {
  width: 100%;
}

.articles-container .ant-typography {
  margin-bottom: 0;
}

.articles-container .ant-row {
  width: 100%;
}

.articles-container .ant-pagination {
  margin-top: 48px;
  text-align: center;
}

/* Removed .second-article-col style: now handled by ArticleCard modifier class */

@media (max-width: 768px) {
  .articles-container {
    padding: 16px;
  }
  
  .articles-container .ant-space {
    gap: 16px !important;
  }
}

@media (max-width: 480px) {
  .articles-container {
    padding: 12px;
  }
}
`; 