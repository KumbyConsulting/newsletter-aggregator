'use client';

import React from 'react';
import { Tooltip, Typography, Card, Empty, Divider, Badge } from 'antd';

const { Text, Title } = Typography;

interface ChartDataItem {
  name: string;
  value: number;
  percentage: number;
  fill: string;
}

interface PieChartProps {
  data: ChartDataItem[];
  size?: number;
}

const PieChart: React.FC<PieChartProps> = ({ 
  data, 
  size = 300 
}) => {
  if (!data || data.length === 0) {
    return (
      <Card className="chart-card">
        <Empty description="No data available for chart visualization" />
      </Card>
    );
  }

  const total = data.reduce((sum, item) => sum + item.value, 0);
  
  // Calculate segments for the pie chart
  let cumulativePercentage = 0;
  const segments = data.map((item, index) => {
    const itemPercentage = (item.value / total) * 100;
    const startPercentage = cumulativePercentage;
    cumulativePercentage += itemPercentage;
    
    // CSS conic-gradient uses degrees, convert from percentage
    const startAngle = startPercentage * 3.6; // 360 degrees / 100%
    const endAngle = cumulativePercentage * 3.6;
    
    return {
      ...item,
      startAngle,
      endAngle,
      itemPercentage
    };
  });
  
  // Generate conic-gradient CSS for the pie chart
  const generatePieChartBackground = () => {
    let gradientString = '';
    segments.forEach((segment, index) => {
      if (index > 0) gradientString += ', ';
      gradientString += `${segment.fill} ${segment.startAngle}deg ${segment.endAngle}deg`;
    });
    return `conic-gradient(${gradientString})`;
  };

  return (
    <Card className="chart-card" bordered={false}>
      <Title level={5} className="chart-title">Topic Distribution</Title>
      
      <div className="pie-chart-content">
        <div className="pie-chart-visual">
          <div 
            className="pie-chart"
            style={{
              width: `${size}px`,
              height: `${size}px`,
              borderRadius: '50%',
              background: generatePieChartBackground(),
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)'
            }}
          >
            <div className="pie-chart-center">
              <Title level={3}>{total}</Title>
              <Text>Total Articles</Text>
            </div>
          </div>
        </div>
        
        <Divider orientation="left">
          <Text type="secondary">Legend</Text>
        </Divider>
        
        <div className="pie-chart-legend">
          {segments.map((segment, index) => (
            <Tooltip 
              key={index}
              title={
                <div className="chart-tooltip">
                  <div className="tooltip-title">{segment.name}</div>
                  <div className="tooltip-item">
                    <span>Count:</span> 
                    <span className="tooltip-value">{segment.value}</span>
                  </div>
                  <div className="tooltip-item">
                    <span>Percentage:</span>
                    <span className="tooltip-value">{segment.itemPercentage.toFixed(1)}%</span>
                  </div>
                </div>
              }
            >
              <div className="legend-item">
                <Badge 
                  color={segment.fill} 
                  text={
                    <div className="legend-text">
                      <Text ellipsis={{ tooltip: segment.name }} className="legend-label">
                        {segment.name}
                      </Text>
                      <Text type="secondary" className="legend-value">
                        {segment.value} ({segment.itemPercentage.toFixed(1)}%)
                      </Text>
                    </div>
                  } 
                />
              </div>
            </Tooltip>
          ))}
        </div>
      </div>
    </Card>
  );
};

export default PieChart; 