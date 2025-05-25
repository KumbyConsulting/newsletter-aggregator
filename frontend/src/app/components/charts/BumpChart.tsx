'use client';
import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import type { TopicTrendSeries } from '@/services/api';

interface BumpChartProps {
  trendData: TopicTrendSeries[];
  height?: number;
}

const COLORS = [
  '#1677ff', '#52c41a', '#faad14', '#ff4d4f', '#722ed1', '#13c2c2', '#fa8c16', '#eb2f96',
  '#a0d911', '#1890ff', '#f5222d', '#fa541c', '#fadb14', '#52c41a', '#1677ff',
];

function transformToRanks(trendData: TopicTrendSeries[]) {
  if (!trendData || trendData.length === 0) return [];
  const allDates = Array.from(new Set(trendData.flatMap(t => t.series.map(d => d.date)))).sort();
  // For each date, rank topics by count (1=top)
  return allDates.map(date => {
    const counts = trendData.map(t => ({ topic: t.topic, count: t.series.find(s => s.date === date)?.count || 0 }));
    counts.sort((a, b) => b.count - a.count);
    const row: any = { date };
    counts.forEach((c, i) => {
      row[c.topic] = c.count > 0 ? i + 1 : null; // null for missing
    });
    return row;
  });
}

const BumpChart: React.FC<BumpChartProps> = ({ trendData, height = 400 }) => {
  const data = transformToRanks(trendData);
  const topics = trendData.map(t => t.topic);
  if (!data.length || !topics.length) {
    return <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>No trend data</div>;
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 24, right: 24, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis reversed allowDecimals={false} tick={{ fontSize: 12 }} domain={[1, topics.length]} />
        <Tooltip />
        <Legend />
        {topics.map((topic, i) => (
          <Line
            key={topic}
            type="monotone"
            dataKey={topic}
            stroke={COLORS[i % COLORS.length]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default BumpChart; 