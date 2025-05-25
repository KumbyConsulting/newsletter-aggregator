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

  return (
    <Card className="chart-card" bordered={false}>
      <Title level={5} className="chart-title">Topic Distribution</Title>
      <div
        style={{ height: height - 80, position: 'relative', marginTop: 20 }}
        role="region"
        aria-label="Bar chart of topic distribution"
      >
        <div className="chart-y-axis">
          {[...Array(5)].map((_, i) => {
            const value = Math.round(maxValue * (4 - i) / 4);
            return (
              <div key={i} className="chart-y-label">
                <Text type="secondary">{value}</Text>
              </div>
            );
          })}
        </div>
        <div className="bar-chart-container">
          {data.map((item, index) => {
            const barHeight = (item.value / maxValue) * (height - 160);
            return (
              <div key={index} className="bar-chart-column">
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
                      }}
                      role="img"
                      aria-label={`${item.name}: ${item.value} articles (${item.percentage.toFixed(1)}%)`}
                    />
                  </div>
                </Tooltip>
                <Text
                  className="bar-chart-label"
                  ellipsis={{ tooltip: item.name }}
                >
                  {item.name}
                </Text>
              </div>
            );
          })}
        </div>
        <div className="chart-x-axis">
          <Text type="secondary">Topics</Text>
        </div>
      </div>
    </Card>
  );
};

export default BarChart; 