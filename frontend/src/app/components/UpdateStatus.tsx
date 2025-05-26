'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  Card, 
  Typography, 
  Button, 
  Progress, 
  Space, 
  message, 
  Statistic, 
  Flex,
  Descriptions,
  Divider,
  Alert
} from 'antd';
import { 
  SyncOutlined, 
  CloseOutlined, 
  CheckCircleOutlined, 
  WarningOutlined, 
  CloseCircleOutlined,
  ReloadOutlined,
  BarChartOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { 
  getUpdateStatus, 
  startUpdate,
  getScrapingMetrics,
  UpdateStatus as UpdateStatusType, 
  ScrapingMetrics 
} from '@/services/api';

const { Title, Text } = Typography;

interface UpdateStatusProps {
  visible: boolean;
  onClose: () => void;
}

export default function UpdateStatus({ visible, onClose }: UpdateStatusProps) {
  const [statusData, setStatusData] = useState<UpdateStatusType>({
    in_progress: false,
    status: 'idle',
    progress: 0,
    message: 'Ready to update',
    sources_processed: 0,
    total_sources: 0,
    articles_found: 0,
    last_update: null,
    error: null,
    estimated_completion_time: null,
    can_be_cancelled: false
  });
  const [metrics, setMetrics] = useState<ScrapingMetrics | null>(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!visible) return;

    // Helper to connect WebSocket
    const connectWS = () => {
      // Use correct protocol for current page
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const backendHost = process.env.NEXT_PUBLIC_BACKEND_WS_HOST || 'newsletter-aggregator-857170198287.us-central1.run.app';
      const wsUrl = `${protocol}://${backendHost}/ws/status`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setStatusData(data);
        } catch (e) {
          // Ignore parse errors
        }
      };
      ws.onerror = () => {
        ws.close();
      };
      ws.onclose = () => {
        wsRef.current = null;
        // Attempt to reconnect after 2 seconds if still visible
        if (visible && !reconnectTimeout.current) {
          reconnectTimeout.current = setTimeout(() => {
            reconnectTimeout.current = null;
            connectWS();
          }, 2000);
        }
      };
    };

    connectWS();
    fetchScrapingMetrics();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }
    };
  }, [visible]);

  const fetchScrapingMetrics = async () => {
    try {
      const metricsData = await getScrapingMetrics();
      setMetrics(metricsData);
    } catch (error) {
      console.error('Error fetching scraping metrics:', error);
    }
  };

  const handleStartUpdate = async () => {
    try {
      setIsLoading(true);
      const result = await startUpdate();
      
      if (result.success) {
        message.success('Update process started');
        setStatusData(result.status);
      } else {
        message.error(result.message || 'Failed to start update');
      }
    } catch (error) {
      console.error('Error starting update:', error);
      message.error('Failed to start update process');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMetrics = () => {
    setShowMetrics(!showMetrics);
    if (!showMetrics && !metrics) {
      fetchScrapingMetrics();
    }
  };

  if (!visible) return null;

  // Determine status icon and colors
  const getStatusIcon = () => {
    if (statusData.in_progress) {
      return <SyncOutlined spin />;
    }
    
    switch (statusData.status) {
      case 'completed':
        return <CheckCircleOutlined />;
      case 'failed':
        return <CloseCircleOutlined />;
      case 'completed_with_warnings':
      case 'completed_with_errors':
        return <WarningOutlined />;
      default:
        return <InfoCircleOutlined />;
    }
  };

  const getStatusColor = () => {
    if (statusData.in_progress) return '#1890ff'; // blue
    
    switch (statusData.status) {
      case 'completed':
        return '#52c41a'; // green
      case 'failed':
        return '#f5222d'; // red
      case 'completed_with_warnings':
      case 'completed_with_errors':
        return '#faad14'; // yellow
      default:
        return '#1890ff'; // blue
    }
  };

  const getStatusTitle = () => {
    if (statusData.in_progress) return 'Updating News';
    
    switch (statusData.status) {
      case 'completed':
        return 'Update Complete';
      case 'failed':
        return 'Update Failed';
      case 'completed_with_warnings':
      case 'completed_with_errors':
        return 'Update Completed with Warnings';
      case 'idle':
        return 'Ready to Update';
      default:
        return 'Update Status';
    }
  };

  const formatLastUpdate = () => {
    if (!statusData.last_update) return 'Never';
    
    const date = new Date(statusData.last_update * 1000);
    return date.toLocaleString();
  };

  return (
    <Card style={{ marginBottom: 24 }}>
      <Flex justify="space-between" align="center">
        <Space>
          <Title 
            level={4} 
            style={{ 
              margin: 0, 
              color: getStatusColor() 
            }}
          >
            {getStatusIcon()} {getStatusTitle()}
          </Title>
        </Space>
        <Space>
          {!statusData.in_progress && (
            <Button 
              type="primary" 
              icon={<ReloadOutlined />} 
              onClick={handleStartUpdate}
              loading={isLoading}
              disabled={statusData.in_progress}
            >
              Start Update
            </Button>
          )}
          <Button 
            type="text" 
            icon={<BarChartOutlined />} 
            onClick={toggleMetrics}
            aria-label="Toggle metrics"
          />
          <Button 
            type="text" 
            icon={<CloseOutlined />} 
            onClick={onClose}
            aria-label="Close"
          />
        </Space>
      </Flex>

      {statusData.error && (
        <Alert 
          message="Error" 
          description={statusData.error}
          type="error" 
          style={{ marginTop: 12 }}
          showIcon
        />
      )}

      <Text type="secondary" style={{ display: 'block', margin: '12px 0' }}>
        {statusData.message}
      </Text>

      <Progress 
        percent={statusData.progress} 
        status={
          statusData.status === 'failed' 
            ? 'exception' 
            : statusData.status === 'completed' 
              ? 'success' 
              : 'active'
        }
        strokeColor={getStatusColor()}
      />

      <Flex justify="space-between" style={{ marginTop: 12 }}>
        <Statistic 
          title="Sources Processed" 
          value={`${statusData.sources_processed} of ${statusData.total_sources}`} 
          valueStyle={{ fontSize: 14 }}
        />
        <Statistic 
          title="Articles Found" 
          value={statusData.articles_found}
          valueStyle={{ fontSize: 14 }}
        />
        <Statistic 
          title="Last Update" 
          value={formatLastUpdate()}
          valueStyle={{ fontSize: 14 }}
        />
      </Flex>

      {showMetrics && metrics && (
        <>
          <Divider style={{ margin: '16px 0' }} />
          <Title level={5}>Last Update Metrics</Title>
          <Descriptions size="small" column={{ xs: 1, sm: 2, md: 3 }} bordered>
            <Descriptions.Item label="Duration">{metrics.duration_seconds} seconds</Descriptions.Item>
            <Descriptions.Item label="Feeds">{metrics.total_feeds}</Descriptions.Item>
            <Descriptions.Item label="Success Rate">{Math.round((metrics.successful_feeds / metrics.total_feeds) * 100)}%</Descriptions.Item>
            <Descriptions.Item label="Articles">{metrics.total_articles}</Descriptions.Item>
            <Descriptions.Item label="Matched">{metrics.matched_articles}</Descriptions.Item>
            <Descriptions.Item label="Rate Limits">{metrics.rate_limits}</Descriptions.Item>
            <Descriptions.Item label="Cache Hit Rate">{metrics.cache_hit_rate}</Descriptions.Item>
            <Descriptions.Item label="Summary Success">{metrics.summary_success_rate}</Descriptions.Item>
          </Descriptions>
        </>
      )}
    </Card>
  );
} 