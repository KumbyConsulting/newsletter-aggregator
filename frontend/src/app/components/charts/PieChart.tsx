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

  // Responsive size for mobile
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 600;
  const chartSize = isMobile ? 180 : size;

  return (
    <Card className="chart-card" bordered={false} style={{ overflowX: 'auto' }}>
      <Title level={5} className="chart-title" style={{ fontSize: isMobile ? 16 : undefined }}>Topic Distribution</Title>
      <div className="pie-chart-content" style={{ flexDirection: isMobile ? 'column' : 'row', alignItems: 'center' }}>
        <div className="pie-chart-visual">
          <div 
            className="pie-chart"
            style={{
              width: `${chartSize}px`,
              height: `${chartSize}px`,
              borderRadius: '50%',
              background: generatePieChartBackground(),
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)'
            }}
          >
            <div className="pie-chart-center" style={{ fontSize: isMobile ? 12 : undefined }}>
              <Title level={3} style={{ fontSize: isMobile ? 18 : undefined }}>{total}</Title>
              <Text style={{ fontSize: isMobile ? 12 : undefined }}>Total Articles</Text>
            </div>
          </div>
        </div>
        <Divider orientation="left">
          <Text type="secondary" style={{ fontSize: isMobile ? 12 : undefined }}>Legend</Text>
        </Divider>
        <div className="pie-chart-legend" style={{ fontSize: isMobile ? 12 : undefined }}>
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
                      <Text ellipsis={{ tooltip: segment.name }} className="legend-label" style={{ fontSize: isMobile ? 12 : undefined }}>
                        {segment.name}
                      </Text>
                      <Text type="secondary" className="legend-value" style={{ fontSize: isMobile ? 12 : undefined }}>
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