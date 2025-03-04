'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  Typography, 
  Row, 
  Col, 
  Skeleton, 
  Alert, 
  Empty, 
  Button, 
  Space, 
  Tag, 
  Spin, 
  Divider,
  Card,
  Pagination,
  Grid,
  message,
  notification,
  Badge
} from 'antd';
import { 
  SearchOutlined, 
  ReloadOutlined, 
  ExclamationCircleOutlined,
  LeftOutlined,
  RightOutlined,
  LoadingOutlined,
  SyncOutlined,
  NotificationOutlined
} from '@ant-design/icons';
import ArticleCard, { ArticleCardSkeleton } from './ArticleCard';
import { getArticles, Article, getUpdateStatus } from '../services/api';

const { Title, Text, Paragraph } = Typography;
const { useBreakpoint } = Grid;

interface ArticlesGridProps {
  searchQuery?: string;
  selectedTopic?: string;
}

export default function ArticlesGrid({ searchQuery = '', selectedTopic = 'All' }: ArticlesGridProps) {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(9);
  const [totalPages, setTotalPages] = useState(1);
  const [totalArticles, setTotalArticles] = useState(0);
  const [topics, setTopics] = useState<string[]>([]);
  const [isPolling, setIsPolling] = useState(false);
  const [newArticlesAvailable, setNewArticlesAvailable] = useState(false);
  const [lastUpdateTime, setLastUpdateTime] = useState<number | null>(null);
  const pollingInterval = useRef<NodeJS.Timeout | null>(null);
  const screens = useBreakpoint();

  // Fetch articles when search query, topic, or page changes
  useEffect(() => {
    fetchArticles();
  }, [searchQuery, selectedTopic, currentPage, pageSize]);

  // Setup polling for new articles
  useEffect(() => {
    startPolling();
    
    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
    };
  }, [lastUpdateTime]);

  const startPolling = () => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
    }
    
    setIsPolling(true);
    pollingInterval.current = setInterval(checkForNewArticles, 30000); // Poll every 30 seconds
  };

  const stopPolling = () => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
    setIsPolling(false);
  };

  const checkForNewArticles = async () => {
    try {
      // Check update status to see if there are new articles
      const status = await getUpdateStatus();
      
      // If we have a completed update that's newer than our last fetch
      if (status.last_update && (!lastUpdateTime || status.last_update > lastUpdateTime)) {
        setNewArticlesAvailable(true);
        // Show notification
        notification.info({
          message: 'New Articles Available',
          description: 'Refresh to see the latest articles.',
          btn: (
            <Button type="primary" size="small" onClick={refreshArticles}>
              Refresh Now
            </Button>
          ),
          duration: 0, // Don't auto-dismiss
        });
      }
    } catch (error) {
      console.error('Error checking for updates:', error);
    }
  };

  const refreshArticles = () => {
    setNewArticlesAvailable(false);
    // Close any open notifications
    notification.destroy();
    // Reset to first page and fetch
    setCurrentPage(1);
    fetchArticles(true);
  };

  const fetchArticles = async (isRefresh = false) => {
    try {
      setLoading(true);
      
      if (isRefresh) {
        message.loading('Refreshing articles...', 0.5);
      }
      
      console.log('Fetching articles, page:', currentPage, 'size:', pageSize, 'topic:', selectedTopic);
      
      const data = await getArticles(
        currentPage,
        pageSize,
        searchQuery,
        selectedTopic,
        'pub_date',
        'desc'
      );
      
      console.log('Articles data received:', data);
      
      setArticles(data.articles);
      setTotalPages(data.total_pages);
      setTotalArticles(data.total);
      setTopics(Array.isArray(data.topics) ? data.topics : []);
      setError(null);
      
      // Update last update time to track future updates
      const status = await getUpdateStatus();
      if (status.last_update) {
        setLastUpdateTime(status.last_update);
      }
      
      if (isRefresh) {
        message.success('Articles refreshed successfully!');
      }
    } catch (err) {
      console.error('Error fetching articles:', err);
      setError(`Error loading articles: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle page change
  const handlePageChange = (page: number, size?: number) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (size && size !== pageSize) {
      setPageSize(size);
    }
  };

  // Loading state - show skeleton cards
  if (loading && articles.length === 0) {
    return (
      <Row gutter={[24, 24]}>
        {[...Array(pageSize)].map((_, i) => (
          <Col xs={24} md={12} lg={8} key={i}>
            <ArticleCardSkeleton />
          </Col>
        ))}
      </Row>
    );
  }

  // Error state
  if (error) {
    return (
      <Alert
        message="Error"
        description={error}
        type="error"
        showIcon
        action={
          <Button type="primary" onClick={refreshArticles}>
            Try Again
          </Button>
        }
      />
    );
  }

  // No articles found
  if (articles.length === 0) {
    return (
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={
          <Space direction="vertical" align="center">
            <Title level={4}>No articles found</Title>
            <Paragraph type="secondary">
              {selectedTopic !== 'All' || searchQuery ? 
                "Try adjusting your search or filter to find what you're looking for." :
                "No articles are currently available. Try running the article update process."}
            </Paragraph>
          </Space>
        }
      >
        <Space>
          {(selectedTopic !== 'All' || searchQuery) && (
            <Button 
              type="primary" 
              onClick={() => window.location.href = '/'} 
              icon={<SearchOutlined />}
            >
              Clear Filters
            </Button>
          )}
          <Button 
            onClick={refreshArticles} 
            icon={<ReloadOutlined />}
          >
            Refresh
          </Button>
        </Space>
      </Empty>
    );
  }

  return (
    <div className="articles-grid">
      {/* Header with info */}
      <div style={{ marginBottom: 24 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space align="center">
              <Title level={2} style={{ margin: 0 }}>
                {selectedTopic !== 'All' ? selectedTopic : 'Latest'} Articles
              </Title>
              {newArticlesAvailable && (
                <Badge dot>
                  <Button 
                    type="primary" 
                    icon={<SyncOutlined />} 
                    onClick={refreshArticles}
                    shape="round"
                    size="small"
                  >
                    New Articles Available
                  </Button>
                </Badge>
              )}
            </Space>
          </Col>
          <Col>
            <Space>
              <Button 
                icon={<ReloadOutlined spin={isPolling} />}
                onClick={isPolling ? stopPolling : startPolling}
                type={isPolling ? "default" : "primary"}
              >
                {isPolling ? "Auto Refresh On" : "Auto Refresh Off"}
              </Button>
              <Text type="secondary">
                Showing {(currentPage - 1) * pageSize + 1}-{Math.min(currentPage * pageSize, totalArticles)} of {totalArticles} articles
              </Text>
            </Space>
          </Col>
        </Row>
        
        {/* Active Filters Display */}
        {(selectedTopic !== 'All' || searchQuery) && (
          <Space style={{ marginTop: 12 }} wrap>
            {selectedTopic !== 'All' && (
              <Tag color="blue" closable onClose={() => window.location.href = '/'}>
                Topic: {selectedTopic}
              </Tag>
            )}
            
            {searchQuery && (
              <Tag color="blue" closable onClose={() => window.location.href = '/'}>
                Search: {searchQuery}
              </Tag>
            )}
          </Space>
        )}

        {lastUpdateTime && (
          <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
            Last updated: {new Date(lastUpdateTime * 1000).toLocaleString()}
          </Text>
        )}

        <Divider style={{ margin: '16px 0' }} />
      </div>
      
      {/* Articles Grid with loading overlay */}
      <div style={{ position: 'relative' }}>
        {loading && articles.length > 0 && (
          <div style={{ 
            position: 'absolute', 
            zIndex: 1, 
            width: '100%', 
            height: '100%', 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            background: 'rgba(255, 255, 255, 0.7)' 
          }}>
            <Spin 
              indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} 
              tip="Loading articles..."
              size="large"
            />
          </div>
        )}
        
        <Row gutter={[24, 24]}>
          {articles.map((article) => (
            <Col xs={24} md={12} lg={8} key={article.id || Math.random().toString()}>
              <ArticleCard
                id={article.id}
                title={article.title}
                description={article.description}
                link={article.link}
                pub_date={article.pub_date}
                source={article.source}
                topic={article.topic}
                summary={article.summary}
                image_url={article.image_url}
                metadata={article.metadata}
              />
            </Col>
          ))}
        </Row>
      </div>
      
      {/* Pagination */}
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: 32 }}>
        <Pagination
          current={currentPage}
          total={totalArticles}
          pageSize={pageSize}
          onChange={handlePageChange}
          showSizeChanger
          pageSizeOptions={['9', '18', '36', '72']}
          responsive
          showTotal={(total, range) => `${range[0]}-${range[1]} of ${total} articles`}
          itemRender={(page, type, originalElement) => {
            if (type === 'prev') {
              return <Button type="text" icon={<LeftOutlined />}>Previous</Button>;
            }
            if (type === 'next') {
              return <Button type="text" icon={<RightOutlined />}>Next</Button>;
            }
            return originalElement;
          }}
          disabled={loading}
        />
      </div>
    </div>
  );
} 