'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  Layout, 
  Typography, 
  Button, 
  Card, 
  Row, 
  Col, 
  Alert, 
  Space, 
  Divider,
  Grid,
  message,
  Statistic,
  Tooltip,
  Badge
} from 'antd';
import { 
  SyncOutlined,
  AppstoreOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  FileTextOutlined,
  CloudDownloadOutlined,
  WarningOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import Header from './components/Header';
import FilterBar from './components/FilterBar';
import ArticlesGrid from './components/ArticlesGrid';
import TopicDistribution from './components/TopicDistribution';
import UpdateStatus from './components/UpdateStatus';
import AIInsights from './components/AIInsights';
import { getTopicStats, getUpdateStatus, startUpdate, TopicStats, getScrapingMetrics, ScrapingMetrics } from './services/api';

const { Content, Footer } = Layout;
const { Title, Text, Paragraph } = Typography;
const { useBreakpoint } = Grid;

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTopic, setSelectedTopic] = useState('All');
  const [showUpdateStatus, setShowUpdateStatus] = useState(false);
  const [topicStats, setTopicStats] = useState<TopicStats[]>([]);
  const [isLoadingStats, setIsLoadingStats] = useState(true);
  const [scrapingMetrics, setScrapingMetrics] = useState<ScrapingMetrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState<number>(0);
  const metricsInterval = useRef<NodeJS.Timeout | null>(null);
  const screens = useBreakpoint();

  // Fetch topic statistics and metrics on component mount
  useEffect(() => {
    fetchTopicStats();
    fetchScrapingMetrics();
    
    // Check if update is in progress on page load
    checkUpdateStatus();
    
    // Start polling for metrics
    metricsInterval.current = setInterval(fetchScrapingMetrics, 60000); // Every minute
    
    return () => {
      if (metricsInterval.current) {
        clearInterval(metricsInterval.current);
      }
    };
  }, []);

  const fetchTopicStats = async () => {
    try {
      setIsLoadingStats(true);
      const data = await getTopicStats();
      setTopicStats(data.topics || []);
    } catch (error) {
      console.error('Error fetching topic statistics:', error);
      message.error('Failed to load topic statistics');
    } finally {
      setIsLoadingStats(false);
    }
  };

  const fetchScrapingMetrics = async () => {
    try {
      setMetricsLoading(true);
      const metrics = await getScrapingMetrics();
      setScrapingMetrics(metrics);
    } catch (error) {
      console.error('Error fetching scraping metrics:', error);
    } finally {
      setMetricsLoading(false);
    }
  };
    
  const checkUpdateStatus = async () => {
    try {
      const status = await getUpdateStatus();
      
      if (status.in_progress) {
        setShowUpdateStatus(true);
      }
      
      if (status.last_update) {
        setLastUpdate(new Date(status.last_update * 1000));
      }
    } catch (error) {
      console.error('Error checking update status:', error);
    }
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
  };

  const handleTopicChange = (topic: string) => {
    setSelectedTopic(topic);
  };

  const handleUpdateClick = async () => {
    try {
      message.loading('Starting update process...', 1);
      
      const result = await startUpdate();
      
      if (result.success) {
        message.success('Update process started successfully');
        setShowUpdateStatus(true);
        
        // Refresh metrics after a short delay
        setTimeout(fetchScrapingMetrics, 2000);
        
        // Increment refresh trigger to update insights
        setRefreshTrigger(prev => prev + 1);
      } else {
        message.warning(result.message || 'Update already in progress');
        setShowUpdateStatus(true);
      }
    } catch (error) {
      console.error('Error starting update:', error);
      message.error('Failed to start update. Please try again later.');
    }
  };

  return (
    <Layout>
      <Header onUpdateClick={handleUpdateClick} />
      
      <Content>
        <div style={{ 
          maxWidth: 1200, 
          margin: '0 auto', 
          padding: screens.xs ? '16px 12px' : '24px 16px' 
        }}>
          {showUpdateStatus && (
            <Row gutter={[0, 16]}>
              <Col span={24}>
                <UpdateStatus 
                  visible={showUpdateStatus} 
                  onClose={() => setShowUpdateStatus(false)} 
                />
              </Col>
            </Row>
          )}
          
          <Row gutter={[0, 16]}>
            <Col span={24}>
              <Card>
                <Row justify="space-between" align="middle" gutter={[16, 16]}>
                  <Col xs={24} md={16}>
                    <Title level={2} style={{ marginBottom: 8 }}>Kumby Consulting Newsboard</Title>
                    <Paragraph>
                      Stay updated with the latest industry news, research, and regulatory developments across various business sectors.
                    </Paragraph>
                    {lastUpdate && (
                      <Text type="secondary">
                        <ClockCircleOutlined style={{ marginRight: 4 }} />
                        Last update: {lastUpdate.toLocaleString()}
                      </Text>
                    )}
                  </Col>
                  <Col xs={24} md={8} style={{ textAlign: screens.xs ? 'left' : 'right' }}>
                    <Button 
                      type="primary"
                      icon={<SyncOutlined />}
                      onClick={handleUpdateClick}
                      size={screens.xs ? 'middle' : 'large'}
                    >
                      Update News
                    </Button>
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
          
          {/* Scraping Metrics Dashboard */}
          {scrapingMetrics && (
            <Row gutter={[0, 16]}>
              <Col span={24}>
                <Card title={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Space>
                      <CloudDownloadOutlined />
                      <span>Newsletter Collection Metrics</span>
                    </Space>
                    <Button 
                      type="text" 
                      icon={<SyncOutlined spin={metricsLoading} />} 
                      onClick={fetchScrapingMetrics}
                      size="small"
                    >
                      Refresh
                    </Button>
                  </div>
                }>
                  <Row gutter={[24, 24]}>
                    <Col xs={12} sm={8} md={6}>
                      <Tooltip title="Total number of feeds processed in the last update">
                        <Statistic 
                          title="Total Feeds" 
                          value={scrapingMetrics.total_feeds} 
                          prefix={<CloudDownloadOutlined />}
                        />
                      </Tooltip>
                    </Col>
                    <Col xs={12} sm={8} md={6}>
                      <Tooltip title="Number of feeds that were successfully processed">
                        <Statistic 
                          title="Successful Feeds" 
                          value={scrapingMetrics.successful_feeds} 
                          valueStyle={{ color: '#3f8600' }}
                          prefix={<CheckCircleOutlined />}
                        />
                      </Tooltip>
                    </Col>
                    <Col xs={12} sm={8} md={6}>
                      <Tooltip title="Number of feeds that failed processing">
                        <Statistic 
                          title="Failed Feeds" 
                          value={scrapingMetrics.failed_feeds}
                          valueStyle={{ color: scrapingMetrics.failed_feeds > 0 ? '#cf1322' : undefined }}
                          prefix={<WarningOutlined />}
                        />
                      </Tooltip>
                    </Col>
                    <Col xs={12} sm={8} md={6}>
                      <Tooltip title="Total articles found during the update">
                        <Statistic 
                          title="Articles Found" 
                          value={scrapingMetrics.total_articles}
                          prefix={<FileTextOutlined />}
                        />
                      </Tooltip>
                    </Col>
                    {scrapingMetrics.rate_limits > 0 && (
                      <Col xs={12} sm={8} md={6}>
                        <Tooltip title="Number of rate limit errors encountered">
                          <Statistic 
                            title="Rate Limits" 
                            value={scrapingMetrics.rate_limits}
                            valueStyle={{ color: '#cf1322' }}
                            prefix={<ThunderboltOutlined />}
                          />
                        </Tooltip>
                      </Col>
                    )}
                  </Row>
                </Card>
              </Col>
            </Row>
          )}
          
          <Row gutter={[0, 16]}>
            <Col span={24}>
              <FilterBar 
                onSearch={handleSearch} 
                onTopicChange={handleTopicChange} 
                selectedTopic={selectedTopic} 
              />
            </Col>
          </Row>
          
          <Row gutter={[0, 16]}>
            <Col span={24}>
              <TopicDistribution 
                topicStats={topicStats} 
                isLoading={isLoadingStats} 
              />
            </Col>
          </Row>
          
          <Row gutter={[0, 16]}>
            <Col span={24}>
              <Card>
                <ArticlesGrid 
                  searchQuery={searchQuery} 
                  selectedTopic={selectedTopic} 
                />
              </Card>
            </Col>
          </Row>
          
          <Row gutter={[0, 16]}>
            <Col span={24}>
              <AIInsights refreshTrigger={refreshTrigger} />
            </Col>
          </Row>
        </div>
      </Content>
      
      <Footer style={{ padding: screens.xs ? '16px 12px' : '24px 16px' }}>
        <Row justify="space-between" align="middle" gutter={[16, 16]}>
          <Col xs={24} md={6} style={{ textAlign: screens.xs ? 'center' : 'left' }}>
            <Space>
              <Text strong style={{ fontSize: 18 }}>Kumby</Text>
              <Text style={{ fontSize: 18, fontWeight: 300 }}>Newsboard</Text>
            </Space>
          </Col>
          <Col xs={24} md={12} style={{ textAlign: 'center' }}>
            <Text type="secondary">
              © {new Date().getFullYear()} Kumby Consulting. All rights reserved.
            </Text>
          </Col>
          <Col xs={24} md={6} style={{ textAlign: screens.xs ? 'center' : 'right' }}>
            <Space>
              <Button type="text" href="#" icon={<AppstoreOutlined />} />
              <Button type="text" href="#" icon={<AppstoreOutlined />} />
            </Space>
          </Col>
        </Row>
      </Footer>
    </Layout>
  );
}
