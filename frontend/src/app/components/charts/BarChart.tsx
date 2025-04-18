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
  
  return (
    <Card className="chart-card" bordered={false}>
      <Title level={5} className="chart-title">Topic Distribution</Title>
      
      <div style={{ height: height - 80, position: 'relative', marginTop: 20 }}>
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
            
            // Generate gradient background
            const gradientId = `barGradient-${index}`;
            
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
                  <div className="bar-chart-bar-container">
                    <svg width="0" height="0">
                      <defs>
                        <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor={item.fill} stopOpacity={0.8} />
                          <stop offset="100%" stopColor={item.fill} stopOpacity={0.4} />
                        </linearGradient>
                      </defs>
                    </svg>
                    <div 
                      className="bar-chart-bar" 
                      style={{ 
                        height: `${barHeight}px`,
                        background: `url(#${gradientId})`,
                        backgroundColor: item.fill 
                      }}
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