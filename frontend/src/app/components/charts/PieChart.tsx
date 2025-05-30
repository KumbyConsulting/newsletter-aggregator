'use client';

import React, { useState, useEffect } from 'react';
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
  const [show, setShow] = useState(false);
  useEffect(() => { setShow(true); }, []);

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
    <div style={{ width: '100%', overflowX: 'auto', padding: isMobile ? 0 : 8 }}>
      <Title level={5} className="chart-title" style={{ fontSize: isMobile ? 16 : undefined, textAlign: 'center', marginBottom: 16 }}>Topic Distribution</Title>
      <div className="pie-chart-content" style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', alignItems: 'center', justifyContent: 'center', gap: isMobile ? 12 : 32, width: '100%' }}>
        <div className="pie-chart-visual" style={{ position: 'relative', opacity: show ? 1 : 0, transition: 'opacity 0.8s', margin: '0 auto', maxWidth: chartSize }}>
          <div 
            className="pie-chart"
            style={{
              width: `${chartSize}px`,
              height: `${chartSize}px`,
              borderRadius: '50%',
              background: generatePieChartBackground(),
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
              position: 'relative',
              margin: '0 auto',
              maxWidth: chartSize,
            }}
          >
            <div className="pie-chart-center" style={{ fontSize: isMobile ? 12 : undefined }}>
              <Title level={3} style={{ fontSize: isMobile ? 18 : undefined }}>{total}</Title>
              <Text style={{ fontSize: isMobile ? 12 : undefined }}>Total Articles</Text>
            </div>
            {/* Overlay percentage labels on pie segments if >8% */}
            {segments.map((segment, index) => {
              if (segment.itemPercentage < 8) return null;
              // Calculate label position (middle angle)
              const angle = (segment.startAngle + segment.endAngle) / 2;
              const radians = (angle - 90) * (Math.PI / 180);
              const radius = chartSize / 2 * 0.7;
              const x = chartSize / 2 + radius * Math.cos(radians);
              const y = chartSize / 2 + radius * Math.sin(radians);
              return (
                <Text
                  key={index}
                  style={{
                    position: 'absolute',
                    left: x,
                    top: y,
                    transform: 'translate(-50%, -50%)',
                    fontWeight: 600,
                    fontSize: isMobile ? 10 : 13,
                    color: '#222',
                    pointerEvents: 'none',
                    textShadow: '0 1px 4px #fff',
                  }}
                >
                  {segment.itemPercentage.toFixed(0)}%
                </Text>
              );
            })}
          </div>
        </div>
        <div style={{ minWidth: isMobile ? 0 : 180, maxWidth: 260, margin: isMobile ? '0 auto' : '0', flex: 1 }}>
          <Divider orientation="left">
            <Text type="secondary" style={{ fontSize: isMobile ? 12 : undefined }}>Legend</Text>
          </Divider>
          <div className="pie-chart-legend" style={{ fontSize: isMobile ? 12 : 13, marginBottom: 8 }}>
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
                <div className="legend-item" style={{ marginBottom: 8 }}>
                  <Badge 
                    color={segment.fill} 
                    text={
                      <div className="legend-text">
                        <Text ellipsis={{ tooltip: segment.name }} className="legend-label" style={{ fontSize: isMobile ? 12 : 13 }}>
                          {segment.name}
                        </Text>
                        <Text type="secondary" className="legend-value" style={{ fontSize: isMobile ? 12 : 13 }}>
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
      </div>
    </div>
  );
};

export default PieChart; 