'use client';

import { 
  Typography, 
  Row, 
  Col, 
  Empty, 
  Space,
  Grid,
  Spin,
  Result,
  ConfigProvider,
  Alert
} from 'antd';
import { Article } from '@/types';
import { ArticleCard, ArticleCardSkeleton } from './ArticleCard';
import { SearchOutlined, InboxOutlined, SyncOutlined } from '@ant-design/icons';
import useSyncArticles from '../hooks/useSyncArticles';
import { useEffect } from 'react';
import React from 'react';

const { Title, Paragraph } = Typography;
const { useBreakpoint } = Grid;

// Add a new component to inject the styles
const ArticlesGridStyles = () => {
  useEffect(() => {
    // Only inject the styles once
    if (!document.getElementById('articles-grid-styles')) {
      const styleElement = document.createElement('style');
      styleElement.id = 'articles-grid-styles';
      styleElement.innerHTML = styles;
      document.head.appendChild(styleElement);
    }
    
    // Clean up on unmount
    return () => {
      const styleElement = document.getElementById('articles-grid-styles');
      if (styleElement) {
        styleElement.remove();
      }
    };
  }, []);
  
  return null;
};

// Highlight utility
function highlightText(text: string, query: string): React.ReactNode {
  if (!query) return text;
  const safeQuery = query.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&');
  const regex = new RegExp(`(${safeQuery})`, 'gi');
  const parts = text.split(regex);
  return parts.map((part, i) =>
    regex.test(part) ? <mark key={i}>{part}</mark> : part
  );
}

interface ArticlesGridProps {
  articles: Article[];
  loading?: boolean;
  error?: string;
  emptyMessage?: string;
  syncWithBackend?: boolean;
  currentSearch?: string;
}

export const ArticlesGrid: React.FC<ArticlesGridProps> = ({
  articles,
  loading = false,
  error,
  emptyMessage = 'No articles found',
  syncWithBackend = true,
  currentSearch = '',
}) => {
  const screens = useBreakpoint();
  
  useEffect(() => {
    console.log(`ArticlesGrid received ${articles.length} articles, loading: ${loading}`);
    if (articles.length > 0) {
      console.log('First article:', articles[0].id, articles[0].metadata.title);
    }
  }, [articles, loading]);
  
  // Only use the sync hook if syncWithBackend is true and we have articles
  // This prevents unnecessary API calls when there are no articles
  const { 
    articles: syncedArticles, 
    isSyncing, 
    invalidCount, 
    hasSyncedArticles 
  } = syncWithBackend && articles && articles.length > 0 
    ? useSyncArticles(articles) 
    : { 
      articles: articles, 
      isSyncing: false, 
      invalidCount: 0, 
      hasSyncedArticles: articles && articles.length > 0 
    };

  // CRITICAL FIX: Ensure we always have articles to display even if sync fails
  const displayArticles = syncedArticles && syncedArticles.length > 0 
    ? syncedArticles 
    : articles;

  // Calculate columns based on screen size
  const getSpan = () => {
    if (screens.xxl) return 6;     // 4 columns on xxl screens
    if (screens.xl) return 8;      // 3 columns on xl screens
    if (screens.lg) return 8;      // 3 columns on large screens
    if (screens.md) return 12;     // 2 columns on medium screens
    if (screens.sm) return 12;     // 2 columns on small screens
    return 24;                     // 1 column on extra small screens
  };

  // Show loading state with skeleton cards
  if (loading || isSyncing) {
    return (
      <div className="articles-grid-loading">
        {isSyncing && !loading && (
          <Alert
            message="Syncing articles with database..."
            description="Ensuring all displayed articles exist in the database."
            type="info"
            showIcon
            icon={<SyncOutlined spin />}
            style={{ marginBottom: 16 }}
          />
        )}
        <Row gutter={[8, 8]} className="fade-in-up">
          {[...Array(6)].map((_, index) => (
            <Col key={index} xs={24} sm={12} md={8} xxl={6} style={{ animationDelay: `${0.05 * index}s`, minHeight: 44 }}>
              <ArticleCardSkeleton />
            </Col>
          ))}
        </Row>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <Result
        status="error"
        title="Failed to load articles"
        subTitle={error}
        className="articles-grid-error"
      />
    );
  }

  // Show sync warning if invalid articles were detected
  const showSyncWarning = syncWithBackend && invalidCount > 0;
  
  // Show empty state
  if (!hasSyncedArticles && !displayArticles.length) {
    console.log('No articles to display');
    return (
      <ConfigProvider
        theme={{
          components: {
            Empty: {
              colorTextDisabled: 'rgba(0, 0, 0, 0.25)',
            },
          },
        }}
      >
        {showSyncWarning && (
          <Alert
            message="Some articles were filtered out"
            description={`${invalidCount} articles were removed because they no longer exist in the database.`}
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <Space direction="vertical" align="center" size="large">
              <div className="empty-icon">
                <InboxOutlined style={{ fontSize: 48, color: '#1890ff' }} />
              </div>
              <Space direction="vertical" align="center" size="small">
                <Title level={5} style={{ margin: 0 }}>{emptyMessage}</Title>
                <Paragraph type="secondary" style={{ margin: 0, textAlign: 'center' }}>
                  Try adjusting your search criteria or filters
                </Paragraph>
              </Space>
            </Space>
          }
          className="articles-grid-empty"
        />
      </ConfigProvider>
    );
  }

  // Show articles grid with sync warning if needed
  console.log(`Rendering ${displayArticles.length} articles grid`);
  return (
    <div className="articles-grid" style={{ overflowX: 'auto' }}>
      <ArticlesGridStyles />
      {showSyncWarning && (
        <Alert
          message="Some articles were filtered out"
          description={`${invalidCount} articles were removed because they no longer exist in the database.`}
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}
      <Row gutter={[8, 8]} className="fade-in-up">
        {displayArticles.map((article, index) => (
          <Col
            key={article.id}
            xs={24}
            sm={12}
            md={8}
            xxl={6}
            style={{
              animationDelay: `${0.05 * index}s`,
              minHeight: 44,
            }}
          >
            <ArticleCard 
              article={article} 
              highlight={currentSearch}
              highlighted={index === 1}
            />
          </Col>
        ))}
      </Row>
    </div>
  );
};

// Add styles to your global CSS or a separate module
const styles = `
.articles-grid {
  min-height: 200px;
  padding: 12px 0;
}

.articles-grid .ant-row {
  margin: -16px -16px;
}

.articles-grid .ant-col {
  padding: 16px 16px;
}

.articles-grid-loading {
  min-height: 400px;
  padding: 32px 0;
  background: #fafafa;
  border-radius: 12px;
}

.articles-grid-error {
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--background-light, #fafafa);
  border-radius: 12px;
  margin: 24px 0;
  padding: 32px;
}

.articles-grid-empty {
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px 24px;
  background: var(--background-light, #fafafa);
  border-radius: 12px;
  margin: 24px 0;
  text-align: center;
}

.empty-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 96px;
  height: 96px;
  border-radius: 50%;
  background: var(--background-lighter, #f0f2f5);
  margin-bottom: 24px;
  color: var(--primary-color, #1890ff);
}

.empty-icon .anticon {
  font-size: 40px;
}

.articles-grid-empty .ant-typography {
  max-width: 400px;
  margin-left: auto;
  margin-right: auto;
}

.articles-grid-empty h5.ant-typography {
  font-size: 1.25rem;
  margin-bottom: 12px;
  color: var(--heading-color, #262626);
}

.articles-grid-empty .ant-typography.ant-typography-secondary {
  font-size: 1rem;
  line-height: 1.6;
  color: var(--text-secondary, rgba(0, 0, 0, 0.65));
}

/* Animation for grid items */
.fade-in-up .ant-col {
  animation: fadeInUp 0.5s ease;
  animation-fill-mode: both;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Topic badge styling */
.topic-badge {
  font-weight: 500;
  text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

/* Article Card Image Container */
.article-image-container {
  position: relative;
  width: 100%;
  height: 220px;
  overflow: hidden;
  background-color: #f5f5f5;
}

.article-image-container::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(180deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0) 40%);
  pointer-events: none;
}

.article-image {
  transition: transform 0.5s ease;
}

.article-card:hover .article-image {
  transform: scale(1.05);
}

/* Card Styling */
.article-card {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  height: 100%;
}

.article-card:hover {
  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}

.article-title {
  font-weight: 600;
  transition: color 0.3s ease;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .articles-grid .ant-row {
    margin: -12px -12px;
  }

  .articles-grid .ant-col {
    padding: 12px 12px;
  }

  .articles-grid-empty {
    padding: 32px 16px;
  }

  .empty-icon {
    width: 72px;
    height: 72px;
  }

  .empty-icon .anticon {
    font-size: 32px;
  }

  .article-image-container {
    height: 180px;
  }
}

/* High contrast mode */
@media (prefers-contrast: high) {
  .articles-grid-empty,
  .articles-grid-error,
  .articles-grid-loading {
    background: #ffffff;
    border: 2px solid #000000;
  }
}
`; 