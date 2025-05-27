'use client';

import React from 'react';
import { Typography, Tooltip, Card, Empty } from 'antd';

const { Text, Title } = Typography;

interface ChartDataItem {
  name: string;
  value: number;
  percentage: number;
  fill: string;
}

interface BarChartProps {
  data: ChartDataItem[];
  height?: number;
}

const BarChart: React.FC<BarChartProps> = ({ 
  data, 
  height = 400 
}) => {
  if (!data || data.length === 0) {
    return (
      <Card className="chart-card">
        <Empty description="No data available for chart visualization" />
      </Card>
    );
  }

  // Find max value for scaling
  const maxValue = Math.max(...data.map(item => item.value));
  if (maxValue <= 0) {
    return (
      <Card className="chart-card">
        <Empty description="No positive data to display" />
      </Card>
    );
  }

  // Responsive height for mobile
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 600;
  const chartHeight = isMobile ? 220 : height;

  return (
    <Card className="chart-card" bordered={false} style={{ overflowX: 'auto' }}>
      <Title level={5} className="chart-title" style={{ fontSize: isMobile ? 16 : undefined }}>Topic Distribution</Title>
      <div
        style={{ height: chartHeight - 80, position: 'relative', marginTop: 20 }}
        role="region"
        aria-label="Bar chart of topic distribution"
      >
        <div className="chart-y-axis" style={{ fontSize: isMobile ? 10 : undefined }}>
          {[...Array(5)].map((_, i) => {
            const value = Math.round(maxValue * (4 - i) / 4);
            return (
              <div key={i} className="chart-y-label">
                <Text type="secondary" style={{ fontSize: isMobile ? 10 : undefined }}>{value}</Text>
              </div>
            );
          })}
        </div>
        <div className="bar-chart-container">
          {data.map((item, index) => {
            const barHeight = (item.value / maxValue) * (chartHeight - 160);
            return (
              <div key={index} className="bar-chart-column">
                {item.value > 0 && (
                  <div style={{ fontWeight: 600, fontSize: isMobile ? 10 : 12, marginBottom: 2, textAlign: 'center' }}>{item.value}</div>
                )}
                <Tooltip
                  title={
                    <div className="chart-tooltip">
                      <div className="tooltip-title">{item.name}</div>
                      <div className="tooltip-item">
                        <span>Count:</span>
                        <span className="tooltip-value">{item.value}</span>
                      </div>
                      <div className="tooltip-item">
                        <span>Percentage:</span>
                        <span className="tooltip-value">{item.percentage.toFixed(1)}%</span>
                      </div>
                    </div>
                  }
                >
                  <div
                    className="bar-chart-bar-container"
                    role="presentation"
                  >
                    <div
                      className="bar-chart-bar"
                      style={{
                        height: `${barHeight}px`,
                        backgroundColor: item.fill,
                        minWidth: isMobile ? 16 : 24,
                        boxShadow: '0 4px 12px rgba(24,144,255,0.12)',
                        cursor: 'pointer',
                        transition: 'box-shadow 0.2s',
                      }}
                      role="img"
                      aria-label={`${item.name}: ${item.value} articles (${item.percentage.toFixed(1)}%)`}
                      onClick={() => alert(item.name)}
                    />
                  </div>
                </Tooltip>
                <Text
                  className="bar-chart-label"
                  ellipsis={{ tooltip: item.name }}
                  style={{ fontSize: isMobile ? 10 : 12, fontWeight: 600, marginTop: 2, textAlign: 'center' }}
                >
                  {item.name}
                </Text>
              </div>
            );
          })}
        </div>
        <div className="chart-x-axis" style={{ fontSize: isMobile ? 10 : undefined }}>
          <Text type="secondary">Topics</Text>
        </div>
      </div>
    </Card>
  );
};

export default BarChart; 