'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { 
  Card, 
  Typography, 
  Button, 
  Table, 
  Skeleton, 
  Empty, 
  Progress, 
  Space,
  Badge,
  Tooltip,
  message
} from 'antd';
import {
  BarChartOutlined,
  TableOutlined,
  SyncOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { getTopicStats } from '../services/api';

const { Title, Text } = Typography;

interface TopicStats {
  topic: string;
  count: number;
  percentage: number;
}

interface TopicDistributionProps {
  topicStats: TopicStats[];
  isLoading: boolean;
}

export default function TopicDistribution({ topicStats: initialTopicStats, isLoading: initialLoading }: TopicDistributionProps) {
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table');
  const [localTopicStats, setLocalTopicStats] = useState<TopicStats[]>(initialTopicStats);
  const [isLoading, setIsLoading] = useState<boolean>(initialLoading);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<any>(null);
  const refreshInterval = useRef<NodeJS.Timeout | null>(null);

  // Sync with prop changes
  useEffect(() => {
    if (!autoRefresh) {
      setLocalTopicStats(initialTopicStats);
      setIsLoading(initialLoading);
      if (initialTopicStats.length > 0) {
        setLastRefreshed(new Date());
      }
    }
  }, [initialTopicStats, initialLoading, autoRefresh]);

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh) {
      fetchLatestStats();
      refreshInterval.current = setInterval(fetchLatestStats, 60000); // Refresh every minute
    }

    return () => {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current);
        refreshInterval.current = null;
      }
    };
  }, [autoRefresh]);

  // Initialize or update chart when topicStats change or view mode changes
  useEffect(() => {
    if (viewMode === 'chart' && chartRef.current && !isLoading && localTopicStats.length > 0) {
      const initChart = async () => {
        try {
          // Dynamically import Chart.js to avoid SSR issues
          const { Chart, registerables } = await import('chart.js');
          Chart.register(...registerables);
          
          // Destroy previous chart instance if it exists
          if (chartInstance.current) {
            chartInstance.current.destroy();
          }
          
          // Create new chart
          const ctx = chartRef.current?.getContext('2d');
          if (ctx) {
            // Create a color palette based on our theme
            const colors = [
              'rgba(43, 125, 233, 0.7)', // primary blue
              'rgba(139, 109, 92, 0.7)', // brown
              'rgba(74, 157, 126, 0.7)', // green
              'rgba(230, 215, 195, 0.7)', // beige
              'rgba(233, 196, 106, 0.7)', // yellow
              'rgba(231, 111, 81, 0.7)', // orange
            ];
            
            chartInstance.current = new Chart(ctx, {
              type: 'bar',
              data: {
                labels: localTopicStats.map(t => t.topic),
                datasets: [{
                  label: 'Number of Articles',
                  data: localTopicStats.map(t => t.count),
                  backgroundColor: localTopicStats.map((_, i) => colors[i % colors.length]),
                  borderColor: localTopicStats.map((_, i) => colors[i % colors.length].replace('0.7', '1')),
                  borderWidth: 1,
                  borderRadius: 6,
                  barThickness: 24,
                  maxBarThickness: 32
                }]
              },
              options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    display: false
                  },
                  tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                      size: 14,
                      weight: 'bold'
                    },
                    bodyFont: {
                      size: 13
                    },
                    callbacks: {
                      label: function(context: any) {
                        const topic = localTopicStats[context.dataIndex];
                        return `Count: ${topic.count} (${topic.percentage}%)`;
                      }
                    }
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    grid: {
                      color: 'rgba(0, 0, 0, 0.05)'
                    }
                  },
                  x: {
                    grid: {
                      display: false
                    }
                  }
                }
              }
            });
          }
        } catch (error) {
          console.error('Error initializing chart:', error);
        }
      };
      
      initChart();
    }
    
    // Cleanup function
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [viewMode, localTopicStats, isLoading]);

  const toggleView = () => {
    setViewMode(prev => prev === 'table' ? 'chart' : 'table');
  };

  const toggleAutoRefresh = () => {
    setAutoRefresh(prev => !prev);
    if (!autoRefresh) {
      message.info('Auto-refresh enabled. Topic stats will update every minute.');
    } else {
      message.info('Auto-refresh disabled.');
    }
  };

  const fetchLatestStats = async () => {
    try {
      setIsLoading(true);
      const result = await getTopicStats();
      if (result && result.topics) {
        setLocalTopicStats(result.topics);
        setLastRefreshed(new Date());
      }
    } catch (error) {
      console.error('Error fetching latest topic stats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const manualRefresh = async () => {
    message.loading('Refreshing topic statistics...', 0.5);
    await fetchLatestStats();
    message.success('Topic statistics updated');
  };

  // Render loading skeleton
  if (isLoading && localTopicStats.length === 0) {
    return (
      <Card>
        <Skeleton active paragraph={{ rows: 6 }} />
      </Card>
    );
  }

  // Render empty state
  if (!isLoading && localTopicStats.length === 0) {
    return (
      <Card>
        <Title level={4}>Topic Distribution</Title>
        <Empty description="No topic data available." />
      </Card>
    );
  }

  // Calculate total article count
  const totalArticles = localTopicStats.reduce((sum, topic) => sum + topic.count, 0);

  // Table columns configuration
  const columns = [
    {
      title: 'Topic',
      dataIndex: 'topic',
      key: 'topic',
      render: (text: string) => (
        <Link 
          href={`/?topic=${encodeURIComponent(text)}`}
          style={{ color: 'var(--primary-color)' }}
        >
          {text}
        </Link>
      ),
    },
    {
      title: 'Count',
      dataIndex: 'count',
      key: 'count',
    },
    {
      title: 'Percentage',
      dataIndex: 'percentage',
      key: 'percentage',
      render: (percentage: number) => `${percentage}%`,
    },
    {
      title: 'Distribution',
      key: 'distribution',
      render: (_: any, record: TopicStats) => (
        <Progress 
          percent={record.percentage} 
          showInfo={false} 
          strokeColor="var(--primary-color)"
          size="small"
        />
      ),
    },
  ];

  return (
    <Card>
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space align="center">
            <Title level={4} style={{ margin: 0 }}>Topic Distribution</Title>
            {autoRefresh && (
              <Badge status="processing" />
            )}
          </Space>
          <Space>
            <Tooltip title={autoRefresh ? "Disable auto-refresh" : "Enable auto-refresh"}>
              <Button
                type={autoRefresh ? "default" : "primary"}
                icon={<SyncOutlined spin={autoRefresh} />}
                onClick={toggleAutoRefresh}
              >
                {autoRefresh ? "Auto" : "Auto Refresh"}
              </Button>
            </Tooltip>
            <Button
              onClick={manualRefresh}
              disabled={isLoading}
              icon={<SyncOutlined spin={isLoading} />}
            >
              Refresh
            </Button>
            <Button
              type="default"
              icon={viewMode === 'table' ? <BarChartOutlined /> : <TableOutlined />}
              onClick={toggleView}
            >
              {viewMode === 'table' ? 'Chart View' : 'Table View'}
            </Button>
          </Space>
        </div>
        
        {/* Metadata row */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <Space>
            <Tooltip title="Total number of articles across all topics">
              <Text type="secondary">
                <InfoCircleOutlined style={{ marginRight: 4 }} />
                {totalArticles} total articles
              </Text>
            </Tooltip>
          </Space>
          {lastRefreshed && (
            <Text type="secondary">
              Last updated: {lastRefreshed.toLocaleTimeString()}
            </Text>
          )}
        </div>
        
        {/* Table View */}
        {viewMode === 'table' && (
          <Table 
            dataSource={localTopicStats} 
            columns={columns} 
            rowKey="topic"
            pagination={false}
            size="middle"
            loading={isLoading && localTopicStats.length > 0}
          />
        )}
        
        {/* Chart View */}
        {viewMode === 'chart' && (
          <div style={{ height: '320px', position: 'relative' }}>
            {isLoading && localTopicStats.length > 0 && (
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
                <SyncOutlined spin style={{ fontSize: 24 }} />
              </div>
            )}
            <canvas ref={chartRef}></canvas>
          </div>
        )}
      </Space>
    </Card>
  );
} 