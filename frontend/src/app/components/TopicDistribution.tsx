'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { 
  Card, 
  Typography, 
  Table,
  Empty, 
  Progress, 
  Space,
  List,
  Statistic,
  Tag,
  Row,
  Col,
  Input,
  Radio,
  Segmented,
  Tooltip,
  Divider,
  Spin,
  Flex,
  Button
} from 'antd';
import {
  BarChartOutlined,
  TableOutlined,
  TagOutlined,
  SearchOutlined,
  PieChartOutlined,
  LineChartOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { getTopicStats } from '@/services/api';
import { TopicStat } from '@/types'; // Import TopicStat type from types

// Simple chart implementation - in a real app, you'd use a chart library like Recharts
import dynamic from 'next/dynamic';

// Dynamically import chart components to avoid SSR issues
const DynamicBarChart = dynamic(
  () => import('./charts/BarChart'),
  { ssr: false, loading: () => <div style={{ height: 300, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>Loading chart...</div> }
);

const DynamicPieChart = dynamic(
  () => import('./charts/PieChart'),
  { ssr: false, loading: () => <div style={{ height: 300, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>Loading chart...</div> }
);

const { Title, Text, Paragraph } = Typography;

interface TopicDistributionProps {
  topics: TopicStat[];
}

// Helper function to generate a URL query string for a topic
const getTopicUrlQuery = (topic: string): string => {
  const params = new URLSearchParams();
  if (topic !== 'All') {
    params.set('topic', topic);
  }
  params.set('page', '1'); 
  return params.toString();
};

// View options for the dashboard
type ViewMode = 'table' | 'barChart' | 'pieChart';

export default function TopicDistribution({ topics }: TopicDistributionProps) {
  const [filteredTopics, setFilteredTopics] = useState<TopicStat[]>(topics);
  const [searchText, setSearchText] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [topN, setTopN] = useState<number>(10);
  const [loading, setLoading] = useState(false);
  const [topicStats, setTopicStats] = useState<any>(null);

  useEffect(() => {
    // Fetch topic distribution stats from backend
    const fetchTopicStats = async () => {
      try {
        console.log('Fetching topic stats using API service...');
        // Use the API service function instead of direct fetch
        const data = await getTopicStats();
        console.log('Topic stats API response:', data);
        setTopicStats(data);
      } catch (error) {
        console.error('Error fetching topic stats:', error);
      }
    };

    fetchTopicStats();
  }, []);

  useEffect(() => {
    if (searchText) {
      setLoading(true);
      // Simulate slight delay to show loading state
      setTimeout(() => {
      const filtered = topics.filter(topic => 
        topic.topic.toLowerCase().includes(searchText.toLowerCase())
      );
      setFilteredTopics(filtered);
        setLoading(false);
      }, 300);
    } else {
      setFilteredTopics(topics);
    }
  }, [searchText, topics]);

  if (!topics || topics.length === 0) {
    return (
      <Card className="dashboard-card">
        <Empty 
          image={Empty.PRESENTED_IMAGE_SIMPLE} 
          description={
            <span className="empty-text">No topic data available</span>
          }
        />
      </Card>
    );
  }

  const totalCount = topics.reduce((sum, topic) => sum + topic.count, 0);

  // Sort topics by count (descending) and take top N for chart display
  const topTopics = [...topics]
    .sort((a, b) => b.count - a.count)
    .slice(0, topN);

  // Prepare data for charts with proper color scheme (Ant Design colors)
  const antdColors = [
    '#1677ff', // Primary blue
    '#52c41a', // Success green
    '#faad14', // Warning yellow
    '#ff4d4f', // Error red
    '#722ed1', // Purple
    '#13c2c2', // Cyan
    '#fa8c16', // Orange
    '#eb2f96', // Magenta
    '#a0d911', // Lime
    '#1890ff', // Bright blue
    '#f5222d', // Volcano
    '#fa541c', // Sunset
    '#fadb14', // Gold
    '#52c41a', // Green
    '#1677ff', // Geekblue
  ];

  const chartData = topTopics.map((topic, index) => ({
    name: topic.topic,
    value: topic.count,
    percentage: topic.percentage,
    fill: antdColors[index % antdColors.length]
  }));

  // Render function for visualization based on view mode
  const renderVisualization = () => {
    if (loading) {
      return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
          <Spin size="large" tip="Loading data..." />
        </div>
      );
    }

    switch (viewMode) {
      case 'barChart':
        return <DynamicBarChart data={chartData} />;
      case 'pieChart':
        return <DynamicPieChart data={chartData} />;
      case 'table':
      default:
        return (
          <Table 
            dataSource={filteredTopics}
            rowKey="topic"
            pagination={{ 
              pageSize: 10, 
              showSizeChanger: true, 
              pageSizeOptions: ['10', '20', '50', '100'],
              showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} topics`,
              style: { marginTop: '16px' }
            }}
            className="topic-distribution-table"
            loading={loading}
          >
            <Table.Column 
              title="Topic" 
              dataIndex="topic" 
              key="topic"
              sorter={(a: TopicStat, b: TopicStat) => a.topic.localeCompare(b.topic)}
              render={(text, record: TopicStat) => (
                <Link href={`/?${getTopicUrlQuery(record.topic)}`} passHref>
                  <Text strong className="topic-link" style={{ cursor: 'pointer', color: 'var(--primary-color)' }}>
                    {record.topic}
                  </Text>
                </Link>
              )}
            />
            <Table.Column 
              title="Count" 
              dataIndex="count" 
              key="count"
              align="right"
              sorter={(a: TopicStat, b: TopicStat) => a.count - b.count}
              defaultSortOrder="descend"
              render={(text, record: TopicStat) => (
                <Space direction="vertical" size={0} align="end">
                  <Text>{record.count}</Text>
                  <Text type="secondary" style={{ fontSize: '12px' }}>articles</Text>
                </Space>
              )}
            />
            <Table.Column 
              title="Percentage" 
              dataIndex="percentage" 
              key="percentage"
              align="right"
              sorter={(a: TopicStat, b: TopicStat) => a.percentage - b.percentage}
              render={(text, record: TopicStat) => (
                <Space direction="vertical" size={0} align="end">
                  <Text>{record.percentage.toFixed(1)}%</Text>
                  <Text type="secondary" style={{ fontSize: '12px' }}>of total</Text>
                </Space>
              )}
            />
            <Table.Column 
              title="Distribution" 
              key="distribution"
              render={(text, record: TopicStat) => (
                <Progress 
                  percent={record.percentage} 
                  size="small"
                  showInfo={false}
                  strokeColor={
                    record.percentage > 50 ? 'var(--success-color)' : 
                    record.percentage > 20 ? 'var(--warning-color)' : 
                    'var(--primary-color)' 
                  }
                  trailColor="var(--border-color)"
                  style={{ width: 150 }}
                />
              )}
            />
            <Table.Column 
              title="Action" 
              key="action"
              align="center"
              render={(text, record: TopicStat) => (
                <Link href={`/?${getTopicUrlQuery(record.topic)}`} passHref>
                  <Button type="link" icon={<BarChartOutlined />} size="small">
                    View
                  </Button>
                </Link>
              )}
            />
          </Table>
        );
    }
  };

  // Dashboard Statistics Row with data from backend
  const renderStatistics = () => (
    <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
      <Col xs={24} sm={8}>
        <Card className="stat-card" variant="borderless" style={{ backgroundColor: '#e6f7ff', borderRadius: '8px' }}>
          <Statistic 
            title={<Text type="secondary">Total Topics</Text>} 
            value={topicStats?.total_topics || topics.length} 
            prefix={<TagOutlined style={{ color: '#1890ff' }} />} 
            valueStyle={{ color: '#1890ff' }} 
          />
        </Card>
      </Col>
      <Col xs={24} sm={8}>
        <Card className="stat-card" variant="borderless" style={{ backgroundColor: '#f6ffed', borderRadius: '8px' }}>
          <Statistic 
            title={<Text type="secondary">Total Articles</Text>} 
            value={topicStats?.total_articles || totalCount} 
            prefix={<BarChartOutlined style={{ color: '#52c41a' }} />} 
            valueStyle={{ color: '#52c41a' }} 
          />
        </Card>
      </Col>
      <Col xs={24} sm={8}>
        <Card className="stat-card" variant="borderless" style={{ backgroundColor: '#fffbe6', borderRadius: '8px' }}>
          <Statistic 
            title={<Text type="secondary">Average per Topic</Text>} 
            value={topicStats?.average_per_topic || 
              (topics.length ? (totalCount / topics.length).toFixed(1) : 0)
            } 
            prefix={<LineChartOutlined style={{ color: '#faad14' }} />} 
            valueStyle={{ color: '#faad14' }} 
          />
        </Card>
      </Col>
    </Row>
  );

  return (
    <div className="dashboard-container" style={{ padding: '24px' }}>
      {renderStatistics()}
      
      <Card className="dashboard-card" variant="borderless" style={{ borderRadius: '8px' }}>
        <Flex justify="space-between" align="center" wrap="wrap" gap="middle" style={{ marginBottom: '16px' }}>
          <div>
            <Title level={4} style={{ marginBottom: 4 }}>Topic Distribution</Title>
            <Text type="secondary">
              Insights into {topicStats?.total_topics || topics.length} topics
            </Text>
          </div>
          <Space size="middle" wrap>
            <Input 
              placeholder="Search topics" 
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
              prefix={<SearchOutlined />}
              allowClear
              style={{ width: 200 }}
            />
            <Segmented
              value={viewMode}
              onChange={(value) => setViewMode(value as ViewMode)}
              options={[
                { value: 'table', icon: <TableOutlined /> },
                { value: 'barChart', icon: <BarChartOutlined /> },
                { value: 'pieChart', icon: <PieChartOutlined /> },
              ]}
            />
          </Space>
        </Flex>

        {viewMode !== 'table' && (
          <Flex justify="end" style={{ marginBottom: 16 }}>
            <Space>
              <Text>Show:</Text>
              <Radio.Group 
                value={topN} 
                onChange={(e) => setTopN(e.target.value)}
                optionType="button"
                buttonStyle="solid"
              >
                <Radio.Button value={5}>Top 5</Radio.Button>
                <Radio.Button value={10}>Top 10</Radio.Button>
                <Radio.Button value={20}>Top 20</Radio.Button>
              </Radio.Group>
              <Tooltip title="Showing top topics by article count">
                <InfoCircleOutlined style={{ color: 'var(--text-secondary)', cursor: 'help' }} />
              </Tooltip>
            </Space>
          </Flex>
        )}

        <div style={{ marginTop: '20px' }}>
          {renderVisualization()}
        </div>
        
        {!loading && (
          <Flex justify="start" style={{ marginTop: 24 }}>
            <Text type="secondary">
              Total: {totalCount} articles across {topics.length} topics.
            </Text>
          </Flex>
        )}
      </Card>
    </div>
  );
}

// Add styles to your global CSS or a separate module
const styles = `
.dashboard-container {
  .stat-card {
    height: 100%;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: all 0.3s ease;

    &:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
      transform: translateY(-2px);
    }

    .ant-statistic {
      .ant-statistic-title {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 8px;
      }

      .ant-statistic-content {
        font-size: 1.75rem;
        font-weight: 600;
      }
    }

    .stat-icon {
      font-size: 1.25rem;
      margin-right: 8px;
    }
  }

  .dashboard-card {
    margin-top: 24px;
    border-radius: 12px;
    background: #ffffff;
  }
}

// ... rest of the existing styles ...
`; 