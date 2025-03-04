'use client';

import { useState, useEffect } from 'react';
import { Typography, Card, Alert, Button, Spin, Space } from 'antd';
import { SyncOutlined, BulbOutlined, InfoCircleOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

interface AIInsightsProps {
  refreshTrigger?: number; // Optional prop to trigger refresh when articles are updated
}

export default function AIInsights({ refreshTrigger }: AIInsightsProps) {
  const [insight, setInsight] = useState<string>('');
  const [sourceCount, setSourceCount] = useState<number>(0);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch insights when component mounts or refreshTrigger changes
  useEffect(() => {
    fetchInsights();
  }, [refreshTrigger]);

  const fetchInsights = async () => {
    try {
      setLoading(true);
      setError(null);

      // Query to generate insights based on recent articles
      const query = "Analyze the most recent articles and identify key trends, patterns, or notable developments. Provide one insightful observation with supporting statistics if available.";
      
      const response = await fetch('/api/rag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          use_history: false, // Don't need conversation history for insights
          insight_mode: true  // Signal this is for insights generation
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch insights');
      }

      const data = await response.json();
      
      if (data.status === 'error') {
        throw new Error(data.error || 'Error generating insights');
      }

      // Update state with the response
      setInsight(data.response);
      setSourceCount(data.sources?.length || 0);
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Error fetching insights:', err);
      setError('Unable to generate insights at this time.');
    } finally {
      setLoading(false);
    }
  };

  // Handle manual refresh
  const handleRefresh = () => {
    fetchInsights();
  };

  // Loading state
  if (loading && !insight) {
    return (
      <Card>
        <Title level={3} style={{ marginBottom: 16 }}>
          <BulbOutlined /> AI-Generated Insights
        </Title>
        <div style={{ display: 'flex', justifyContent: 'center', padding: '20px 0' }}>
          <Spin tip="Generating insights..." />
        </div>
      </Card>
    );
  }

  // Error state
  if (error && !insight) {
    return (
      <Card>
        <Title level={3} style={{ marginBottom: 16 }}>
          <BulbOutlined /> AI-Generated Insights
        </Title>
        <Alert
          message="Couldn't Generate Insights"
          description={error}
          type="error"
          action={
            <Button size="small" type="primary" onClick={handleRefresh}>
              Try Again
            </Button>
          }
        />
      </Card>
    );
  }

  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={3} style={{ margin: 0 }}>
          <BulbOutlined /> AI-Generated Insights
        </Title>
        <Button 
          icon={<SyncOutlined spin={loading} />} 
          onClick={handleRefresh}
          disabled={loading}
        >
          Refresh
        </Button>
      </div>
      
      <Alert
        message={
          <Text italic>
            {insight || "No insights available. Try refreshing to generate new insights."}
          </Text>
        }
        description={
          lastUpdated ? 
            `Generated from ${sourceCount} recent articles • Updated ${lastUpdated.toLocaleString()}` : 
            "No recent updates"
        }
        type="info"
        showIcon
        icon={<InfoCircleOutlined />}
      />
    </Card>
  );
} 