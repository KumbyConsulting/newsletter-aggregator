'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { getTopicStats, TopicStats } from '@/services/api';
import {
  Card,
  Row,
  Col,
  Typography,
  Statistic,
  Tag,
  Space,
  Spin,
  Alert,
  Empty,
  Tooltip,
  Button
} from 'antd';
import {
  RiseOutlined,
  FallOutlined,
  MinusOutlined,
  FileProtectOutlined,
  ExperimentOutlined,
  MedicineBoxOutlined,
  SafetyCertificateOutlined,
  FundOutlined,
  ApiOutlined,
  TeamOutlined,
  RocketOutlined,
  ReloadOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

interface TopicDisplayData {
  topic: string;
  count: number;
  description: string;
  percentage: number;
  trend?: 'up' | 'down' | 'stable';
  growth_rate?: number;
  recent_count?: number;
  icon: any;
}

// Pharmaceutical-specific topic descriptions and icons
const topicMetadata: Record<string, { description: string; icon: any }> = {
  'Regulatory': {
    description: 'FDA approvals, regulatory changes, and compliance updates in the pharmaceutical industry.',
    icon: <FileProtectOutlined />
  },
  'Clinical Trials': {
    description: 'Updates on clinical trials, research outcomes, and patient studies.',
    icon: <ExperimentOutlined />
  },
  'Drug Development': {
    description: 'New drug developments, pipeline updates, and therapeutic advancements.',
    icon: <MedicineBoxOutlined />
  },
  'Safety': {
    description: 'Drug safety updates, adverse events, and pharmacovigilance information.',
    icon: <SafetyCertificateOutlined />
  },
  'Market Access': {
    description: 'Market trends, pricing strategies, and commercial insights.',
    icon: <FundOutlined />
  },
  'Manufacturing': {
    description: 'Manufacturing processes, quality control, and supply chain updates.',
    icon: <ApiOutlined />
  },
  'Business': {
    description: 'Mergers, acquisitions, partnerships, and industry business developments.',
    icon: <TeamOutlined />
  },
  'Innovation': {
    description: 'Digital health, AI in pharma, and innovative therapeutic approaches.',
    icon: <RocketOutlined />
  }
};

// Create fallback data when backend is unavailable
const getFallbackTopics = (): TopicDisplayData[] => {
  return Object.entries(topicMetadata).map(([topic, metadata]) => ({
    topic,
    count: 0,
    percentage: 0,
    description: metadata.description,
    icon: metadata.icon,
    trend: 'stable',
    growth_rate: 0,
    recent_count: 0
  }));
};

export default function TopicsPage() {
  const [topics, setTopics] = useState<TopicDisplayData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [usingFallback, setUsingFallback] = useState(false);

  const fetchTopics = async () => {
    try {
      setLoading(true);
      const apiResponse = await getTopicStats();
      
      const displayTopics: TopicDisplayData[] = (apiResponse.topics || []).map((apiTopic: TopicStats) => {
        const metadata = topicMetadata[apiTopic.topic] || {
          description: `Articles related to ${apiTopic.topic}`,
          icon: <FileProtectOutlined />
        };
        return {
          topic: apiTopic.topic,
          count: apiTopic.count,
          percentage: apiTopic.percentage,
          description: metadata.description,
          icon: metadata.icon,
          trend: apiTopic.trend || 'stable',
          growth_rate: apiTopic.growth_rate,
          recent_count: apiTopic.recent_count
        };
      });
      
      if (displayTopics.length === 0) {
        setUsingFallback(true);
        setTopics(getFallbackTopics());
      } else {
        setUsingFallback(false);
        setTopics(displayTopics);
      }
      setError(null);
    } catch (err) {
      console.error('Error fetching topics:', err);
      setError(`Failed to load topics from server. ${err instanceof Error ? err.message : 'Please try again later.'}`);
      setUsingFallback(true);
      setTopics(getFallbackTopics());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTopics();
  }, []);

  const renderTrendIndicator = (trend?: string, growth_rate?: number) => {
    if (!trend || trend === 'stable') return <MinusOutlined style={{ color: '#666' }} />;
    if (trend === 'up') return <RiseOutlined style={{ color: '#52c41a' }} />;
    return <FallOutlined style={{ color: '#f5222d' }} />;
  };

  const renderTopicCard = (topic: TopicDisplayData) => {
    return (
      <Card 
        hoverable 
        className="h-full flex flex-col"
        styles={{ body: { flexGrow: 1 } }}
        actions={[
          <Link href={`/?topic=${encodeURIComponent(topic.topic)}`} key="view">
            View Articles
          </Link>
        ]}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="small">
          <Space align="center">
            {topic.icon}
            <Title level={4} style={{ margin: 0 }}>{topic.topic}</Title>
          </Space>
          
          <Row gutter={16}>
            <Col span={12}>
              <Statistic 
                title="Articles" 
                value={topic.count} 
                suffix={
                  <Tooltip title={`${topic.percentage !== undefined ? topic.percentage.toFixed(1) : 'N/A'}% of total articles`}> 
                    <Tag color="blue">{topic.percentage !== undefined ? topic.percentage.toFixed(1) : 'N/A'}%</Tag> 
                  </Tooltip>
                }
              />
            </Col>
            <Col span={12}>
              <Statistic
                title="Trend"
                value={topic.growth_rate !== undefined ? `${(topic.growth_rate).toFixed(1)}%` : 'N/A'}
                prefix={renderTrendIndicator(topic.trend, topic.growth_rate)} 
                valueStyle={{ color: topic.trend === 'up' ? '#52c41a' : topic.trend === 'down' ? '#f5222d' : '#666' }}
              />
            </Col>
          </Row>
          
          <Text type="secondary" style={{ minHeight: '3em' }}>
            {topic.description}
          </Text>
          
          {topic.recent_count !== undefined && topic.recent_count > 0 && (
            <Tag color="green">
              {topic.recent_count} new articles this month
            </Tag>
          )}
        </Space>
      </Card>
    );
  };

  return (
    <div className="container" style={{ maxWidth: 1200, margin: '0 auto', padding: '0px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          Pharmaceutical Topics Overview
        </Title>
        <Button 
          icon={<ReloadOutlined />} 
          onClick={fetchTopics}
          loading={loading}
          type="primary"
        >
          Refresh
        </Button>
      </div>
      
      {usingFallback && !loading && (
        <Alert
          message="Using Fallback Data" 
          description="The backend service is not responding. Showing default topic categories instead of actual data. Google Cloud Platform integration is not enabled."
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}
      
      {error && !usingFallback && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}
      
      {loading ? (
        <div style={{ textAlign: 'center', padding: '48px' }}>
          <Spin size="large" />
        </div>
      ) : topics.length === 0 ? (
        <Empty description="No topics found. Try updating the feed." />
      ) : (
        <Row gutter={[24, 24]}>
          {topics.map((topic, index) => (
            <Col xs={24} sm={12} lg={8} key={`${topic.topic}-${index}`}>
              {renderTopicCard(topic)}
            </Col>
          ))}
        </Row>
      )}
    </div>
  );
}