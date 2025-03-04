'use client';

import { useState, useEffect } from 'react';
import { 
  Card, 
  Typography, 
  Button, 
  Progress, 
  Space, 
  message, 
  Statistic, 
  Flex
} from 'antd';
import { 
  SyncOutlined, 
  CloseOutlined, 
  CheckCircleOutlined, 
  WarningOutlined, 
  CloseCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

interface UpdateStatusProps {
  visible: boolean;
  onClose: () => void;
}

interface UpdateStatusData {
  in_progress: boolean;
  status: string;
  progress: number;
  message: string;
  sources_processed: number;
  total_sources: number;
  articles_found: number;
  last_update?: number;
}

export default function UpdateStatus({ visible, onClose }: UpdateStatusProps) {
  const [statusData, setStatusData] = useState<UpdateStatusData>({
    in_progress: true,
    status: 'in_progress',
    progress: 0,
    message: 'Initializing update...',
    sources_processed: 0,
    total_sources: 0,
    articles_found: 0
  });

  useEffect(() => {
    if (!visible) return;

    // Initial status check
    fetchUpdateStatus();

    // Set up polling interval
    const intervalId = setInterval(fetchUpdateStatus, 2000);

    // Cleanup function
    return () => {
      clearInterval(intervalId);
    };
  }, [visible]);

  const fetchUpdateStatus = async () => {
    try {
      const response = await fetch('/api/update/status');
      const data = await response.json();
      setStatusData(data);

      // If update is complete, stop polling after showing the result for a few seconds
      if (!data.in_progress && data.status !== 'idle') {
        setTimeout(() => {
          if (data.status === 'completed') {
            message.success('Update completed successfully!');
          } else if (data.status === 'failed') {
            message.error('Update failed. Please try again.');
          } else if (data.status.includes('warnings') || data.status.includes('errors')) {
            message.warning('Update completed with some issues.');
          }
        }, 3000);
      }
    } catch (error) {
      console.error('Error fetching update status:', error);
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
        return <SyncOutlined />;
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
      default:
        return 'Updating News';
    }
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
        <Button 
          type="text" 
          icon={<CloseOutlined />} 
          onClick={onClose}
          aria-label="Close"
        />
      </Flex>

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
      </Flex>
    </Card>
  );
} 