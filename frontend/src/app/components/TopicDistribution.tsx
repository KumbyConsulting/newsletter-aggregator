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
  Button,
  Badge
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
import { getTopicStats, getTopicTrends, TopicTrendSeries } from '@/services/api';
import { TopicStat as BaseTopicStat } from '@/types';

// Simple chart implementation - in a real app, you'd use a chart library like Recharts
import dynamic from 'next/dynamic';
import TrendChart from './TrendChart';

// Dynamically import chart components to avoid SSR issues
const DynamicBarChart = dynamic(
  () => import('./charts/BarChart'),
  { ssr: false, loading: () => <div style={{ height: 300, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>Loading chart...</div> }
);

const DynamicPieChart = dynamic(
  () => import('./charts/PieChart'),
  { ssr: false, loading: () => <div style={{ height: 300, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>Loading chart...</div> }
);

const DynamicStackedAreaChart = dynamic(
  () => import('./charts/AreaChart'),
  { ssr: false, loading: () => <div style={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>Loading Stacked Area Chart...</div> }
);

const DynamicBumpChart = dynamic(
  () => import('./charts/BumpChart'),
  { ssr: false, loading: () => <div style={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>Loading Bump Chart...</div> }
);

const { Title, Text, Paragraph } = Typography;

// Locally extend TopicStat for safety
interface TopicStat extends BaseTopicStat {
  trend?: 'up' | 'down' | 'stable';
  growth_rate?: number;
}

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
type ViewMode = 'table' | 'barChart' | 'pieChart' | 'heatmap' | 'timeline' | 'stackedArea' | 'bumpChart';

// --- Heatmap Component ---
const TopicHeatmap = ({ trendData }: { trendData: TopicTrendSeries[] }) => {
  if (!trendData || trendData.length === 0) return null;
  // Get all unique dates (columns)
  const allDates = Array.from(
    new Set(trendData.flatMap(t => t.series.map(d => d.date)))
  ).sort();
  // Find max count for color scaling
  const maxCount = Math.max(...trendData.flatMap(t => t.series.map(d => d.count)));
  // Color scale: light gray to blue
  const colorScale = (count: number) => {
    if (count === 0) return '#f5f5f5';
    const intensity = Math.round(60 + 180 * (count / (maxCount || 1)));
    return `rgb(${255 - intensity},${255 - intensity},255)`;
  };
  return (
    <div style={{ margin: '32px 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
        <span role="img" aria-label="heatmap" style={{ fontSize: 20, marginRight: 8 }}>🔥</span>
        <strong>Topic Activity Heatmap (last 14 days)</strong>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', minWidth: 600 }} aria-label="Topic activity heatmap">
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600, background: '#fafafa' }}>Topic</th>
              {allDates.map(date => (
                <th key={date} style={{ padding: '4px 6px', fontWeight: 400, fontSize: 12, background: '#fafafa' }}>{date.slice(5)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trendData.map(topic => (
              <tr key={topic.topic}>
                <td style={{ padding: '4px 8px', fontWeight: 500 }}>{topic.topic}</td>
                {allDates.map(date => {
                  const d = topic.series.find(s => s.date === date);
                  const count = d ? d.count : 0;
                  return (
                    <td
                      key={date}
                      style={{
                        width: 22,
                        height: 22,
                        background: colorScale(count),
                        textAlign: 'center',
                        border: '1px solid #eee',
                        fontSize: 11,
                        color: count > (maxCount * 0.7) ? '#222' : '#666',
                        cursor: count > 0 ? 'pointer' : 'default',
                        transition: 'background 0.2s',
                      }}
                      aria-label={`Count for ${topic.topic} on ${date}: ${count}`}
                      title={`${topic.topic} on ${date}: ${count}`}
                    >
                      {count > 0 ? count : ''}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// --- Timeline Component ---
const TopicTimeline = ({ trendData }: { trendData: TopicTrendSeries[] }) => {
  if (!trendData || trendData.length === 0) return null;
  // Get all unique dates (columns)
  const allDates = Array.from(
    new Set(trendData.flatMap(t => t.series.map(d => d.date)))
  ).sort();
  // For each date, find top 2 topics by count
  const dayTopTopics = allDates.map(date => {
    const topicsForDay = trendData.map(t => ({
      topic: t.topic,
      count: t.series.find(s => s.date === date)?.count || 0
    }));
    const sorted = topicsForDay.sort((a, b) => b.count - a.count);
    return { date, top: sorted.slice(0, 2).filter(t => t.count > 0) };
  });
  // Color palette for badges
  const colors = ['#1677ff', '#52c41a', '#faad14', '#ff4d4f', '#722ed1', '#13c2c2', '#fa8c16', '#eb2f96'];
  const topicColor = (topic: string) => colors[Math.abs(topic.split('').reduce((a, c) => a + c.charCodeAt(0), 0)) % colors.length];
  return (
    <div style={{ margin: '32px 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
        <span role="img" aria-label="timeline" style={{ fontSize: 20, marginRight: 8 }}>🕒</span>
        <strong>Topic Activity Timeline (top spikes per day)</strong>
      </div>
      <div style={{ overflowX: 'auto', paddingBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', minHeight: 60 }} aria-label="Topic activity timeline">
          {dayTopTopics.map(({ date, top }) => (
            <div key={date} style={{ minWidth: 48, marginRight: 8, textAlign: 'center' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, minHeight: 36 }}>
                {top.map((t, i) => (
                  <span
                    key={t.topic}
                    style={{
                      display: 'inline-block',
                      background: topicColor(t.topic),
                      color: '#fff',
                      borderRadius: 8,
                      padding: '2px 8px',
                      fontSize: 12,
                      fontWeight: 500,
                      marginBottom: 2,
                      boxShadow: '0 1px 3px rgba(0,0,0,0.04)'
                    }}
                    aria-label={`Top topic: ${t.topic} (${t.count}) on ${date}`}
                    title={`${t.topic} (${t.count}) on ${date}`}
                  >
                    {t.topic} <span style={{ fontWeight: 400, fontSize: 11 }}>({t.count})</span>
                  </span>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>{date.slice(5)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default function TopicDistribution({ topics }: TopicDistributionProps) {
  const [filteredTopics, setFilteredTopics] = useState<TopicStat[]>(topics);
  const [searchText, setSearchText] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('barChart');
  const [topN, setTopN] = useState<number>(10);
  const [loading, setLoading] = useState(false);
  const [topicStats, setTopicStats] = useState<any>(null);
  const [trendData, setTrendData] = useState<TopicTrendSeries[]>([]);
  const [trendLoading, setTrendLoading] = useState(true);
  const [trendError, setTrendError] = useState<string | null>(null);

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

  useEffect(() => {
    // Fetch topic trend series from backend
    const fetchTrends = async () => {
      setTrendLoading(true);
      setTrendError(null);
      try {
        const res = await getTopicTrends();
        setTrendData(res.topics || []);
      } catch (err) {
        setTrendError('Failed to load trend data');
      } finally {
        setTrendLoading(false);
      }
    };
    fetchTrends();
  }, []);

  // Normalize topic names for mapping
  const trendMap = Object.fromEntries(
    trendData.map(t => [t.topic.toLowerCase().trim(), t.series])
  );

  // Trends Section: Use real trend data, fallback for missing fields
  const trendsToShow = topics.slice(0, 5).map(topic => ({
    topic: topic.topic,
    trend: (topic as any).trend ?? 'stable',
    data: trendMap[topic.topic.toLowerCase().trim()] || [],
  }));

  // Top Movers: Use growth_rate and trend, fallback for missing fields
  const sortedByGrowth = [...topics].sort((a, b) => (((b as any).growth_rate ?? 0) - ((a as any).growth_rate ?? 0)));
  const topGainers = sortedByGrowth.slice(0, 3).map(t => ({ ...t, trend: (t as any).trend ?? 'stable', data: trendMap[t.topic.toLowerCase().trim()] || [], growth_rate: (t as any).growth_rate ?? 0 }));
  const topDecliners = sortedByGrowth.slice(-3).map(t => ({ ...t, trend: (t as any).trend ?? 'stable', data: trendMap[t.topic.toLowerCase().trim()] || [], growth_rate: (t as any).growth_rate ?? 0 }));

  // TrendRow subcomponent for DRYness and accessibility
  const TrendRow = ({ topic, trend, data, color, height = 32 }: { topic: string; trend: string; data: { date: string; count: number }[]; color: string; height?: number }) => (
    <div style={{ minWidth: 160, flex: 1 }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
        <Badge status={trend === 'up' ? 'success' : trend === 'down' ? 'error' : 'default'} aria-label={`Trend: ${trend}`} />
        <Text strong style={{ marginLeft: 6 }}>{topic}</Text>
      </div>
      <TrendChart
        data={data.map(d => ({ label: d.date, value: d.count }))}
        color={color}
        height={height}
        // Add aria-label for accessibility
        aria-label={`Trend sparkline for ${topic}`}
      />
      <span style={{ fontSize: 0, height: 0, overflow: 'hidden' }}>{`Trend data for ${topic}: ${data.map(d => `${d.date}: ${d.count}`).join(', ')}`}</span>
    </div>
  );

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
      case 'stackedArea':
        return <DynamicStackedAreaChart trendData={trendData} height={400} />;
      case 'bumpChart':
        return <DynamicBumpChart trendData={trendData} height={400} />;
      case 'heatmap':
        return <TopicHeatmap trendData={trendData} />;
      case 'timeline':
        return <TopicTimeline trendData={trendData} />;
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
      {/* Trends Section */}
      <Card className="dashboard-card" style={{ marginBottom: 24, borderRadius: 12 }}>
        <Title level={4} style={{ marginBottom: 8 }}>Topic Trends</Title>
        {trendLoading ? (
          <Spin size="large" tip="Loading trends..." />
        ) : trendError ? (
          <Text type="danger">{trendError}</Text>
        ) : (
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            {trendsToShow.map((trend, idx) => (
              <TrendRow
                key={trend.topic}
                topic={trend.topic}
                trend={trend.trend}
                data={trend.data}
                color={trend.trend === 'up' ? '#52c41a' : trend.trend === 'down' ? '#ff4d4f' : '#888'}
                height={32}
              />
            ))}
          </div>
        )}
      </Card>

      {/* Top Movers Section */}
      <div style={{ display: 'flex', gap: 24, marginBottom: 24, flexWrap: 'wrap' }}>
        <Card className="stat-card" style={{ background: '#f6ffed', flex: 1, borderRadius: 12 }}>
          <Title level={5} style={{ color: '#52c41a', marginBottom: 8 }}>Top Gainers</Title>
          {topGainers.map((t, i) => (
            <div key={t.topic} style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
              <Badge color="#52c41a" aria-label="Top gainer" />
              <Text strong style={{ marginLeft: 8 }}>{t.topic}</Text>
              <Statistic value={t.count} valueStyle={{ fontSize: 18, marginLeft: 12 }} prefix={<span style={{ color: '#52c41a' }}>{t.growth_rate ? `+${t.growth_rate}%` : ''}</span>} />
              <TrendChart data={t.data.map(d => ({ label: d.date, value: d.count }))} color="#52c41a" height={24} aria-label={`Trend sparkline for ${t.topic}`} />
              <span style={{ fontSize: 0, height: 0, overflow: 'hidden' }}>{`Trend data for ${t.topic}: ${t.data.map(d => `${d.date}: ${d.count}`).join(', ')}`}</span>
            </div>
          ))}
        </Card>
        <Card className="stat-card" style={{ background: '#fff1f0', flex: 1, borderRadius: 12 }}>
          <Title level={5} style={{ color: '#ff4d4f', marginBottom: 8 }}>Top Decliners</Title>
          {topDecliners.map((t, i) => (
            <div key={t.topic} style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
              <Badge color="#ff4d4f" aria-label="Top decliner" />
              <Text strong style={{ marginLeft: 8 }}>{t.topic}</Text>
              <Statistic value={t.count} valueStyle={{ fontSize: 18, marginLeft: 12 }} prefix={<span style={{ color: '#ff4d4f' }}>{t.growth_rate ? `${t.growth_rate}%` : ''}</span>} />
              <TrendChart data={t.data.map(d => ({ label: d.date, value: d.count }))} color="#ff4d4f" height={24} aria-label={`Trend sparkline for ${t.topic}`} />
              <span style={{ fontSize: 0, height: 0, overflow: 'hidden' }}>{`Trend data for ${t.topic}: ${t.data.map(d => `${d.date}: ${d.count}`).join(', ')}`}</span>
            </div>
          ))}
        </Card>
      </div>

      {/* Existing Statistics and Table/Charts */}
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
                { value: 'table', icon: <TableOutlined />, label: 'Table' },
                { value: 'barChart', icon: <BarChartOutlined />, label: 'Bar' },
                { value: 'pieChart', icon: <PieChartOutlined />, label: 'Pie' },
                { value: 'stackedArea', icon: <LineChartOutlined />, label: 'Stacked Area' },
                { value: 'bumpChart', icon: <LineChartOutlined />, label: 'Bump' },
                { value: 'heatmap', icon: <span role="img" aria-label="heatmap">🔥</span>, label: 'Heatmap' },
                { value: 'timeline', icon: <LineChartOutlined />, label: 'Timeline' },
              ]}
              block
            />
          </Space>
        </Flex>

        {['barChart', 'pieChart'].includes(viewMode) && (
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