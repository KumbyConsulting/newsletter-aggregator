import React from 'react';

interface TrendChartProps {
  title?: string;
  data: { label: string; value: number }[];
  color?: string;
  height?: number;
}

const TrendChart: React.FC<TrendChartProps> = ({ title, data, color = '#1677ff', height = 32 }) => {
  if (!data || data.length === 0) return null;
  const width = 100;
  const max = Math.max(...data.map(d => d.value));
  const min = Math.min(...data.map(d => d.value));
  const range = max - min || 1;
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((d.value - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(' ');
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
      {title && <div style={{ fontSize: 12, color: '#888', marginBottom: 2 }}>{title}</div>}
      <svg width={width} height={height} style={{ display: 'block' }}>
        <polyline
          fill="none"
          stroke={color}
          strokeWidth={2}
          points={points}
        />
        {/* Optionally, add a dot for the last value */}
        <circle
          cx={width}
          cy={height - ((data[data.length - 1].value - min) / range) * (height - 4) - 2}
          r={2.5}
          fill={color}
        />
      </svg>
    </div>
  );
};

export default TrendChart; 