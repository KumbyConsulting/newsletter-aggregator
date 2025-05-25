'use client';
import React from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import type { TopicTrendSeries } from '@/services/api';

interface AreaChartProps {
  trendData: TopicTrendSeries[];
  height?: number;
}

const COLORS = [
  '#1677ff', '#52c41a', '#faad14', '#ff4d4f', '#722ed1', '#13c2c2', '#fa8c16', '#eb2f96',
  '#a0d911', '#1890ff', '#f5222d', '#fa541c', '#fadb14', '#52c41a', '#1677ff',
];

function transformData(trendData: TopicTrendSeries[]) {
  if (!trendData || trendData.length === 0) return [];
  // Collect all unique dates
  const allDates = Array.from(new Set(trendData.flatMap(t => t.series.map(d => d.date)))).sort();
  // Build a row for each date
  return allDates.map(date => {
    const row: any = { date };
    trendData.forEach(topic => {
      const found = topic.series.find(s => s.date === date);
      row[topic.topic] = found ? found.count : 0;
    });
    return row;
  });
}

const StackedAreaChart: React.FC<AreaChartProps> = ({ trendData, height = 400 }) => {
  const data = transformData(trendData);
  const topics = trendData.map(t => t.topic);
  if (!data.length || !topics.length) {
    return <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>No trend data</div>;
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 24, right: 24, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip />
        <Legend />
        {topics.map((topic, i) => (
          <Area
            key={topic}
            type="monotone"
            dataKey={topic}
            stackId="1"
            stroke={COLORS[i % COLORS.length]}
            fill={COLORS[i % COLORS.length]}
            fillOpacity={0.7}
            isAnimationActive={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default StackedAreaChart; 