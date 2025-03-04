'use client';

import React, { useState } from 'react';
import { Button, DatePicker, Space, Table, Tag, Input, Select, Card, Statistic, Tabs, Typography } from 'antd';
import { SearchOutlined, FilterOutlined, BarChartOutlined, FileTextOutlined, BellOutlined } from '@ant-design/icons';
import type { TableProps } from 'antd';

const { Title, Paragraph } = Typography;
const { Option } = Select;

// Define interface for article data
interface DataType {
  key: string;
  title: string;
  source: string;
  topic: string;
  date: string;
  tags: string[];
}

// Sample data for the table
const data: DataType[] = [
  {
    key: '1',
    title: 'New Breakthrough in Cancer Treatment Shows Promise',
    source: 'Nature Medicine',
    topic: 'Oncology',
    date: '2023-10-15',
    tags: ['cancer', 'immunotherapy', 'clinical trial'],
  },
  {
    key: '2',
    title: 'FDA Approves Novel Alzheimer\'s Drug',
    source: 'New England Journal of Medicine',
    topic: 'Neurology',
    date: '2023-09-28',
    tags: ['alzheimer', 'neurodegenerative', 'FDA'],
  },
  {
    key: '3',
    title: 'COVID-19 Vaccine Efficacy Against New Variants',
    source: 'The Lancet',
    topic: 'Infectious Disease',
    date: '2023-10-05',
    tags: ['covid-19', 'vaccine', 'variants'],
  },
  {
    key: '4',
    title: 'Advancements in Gene Therapy for Rare Diseases',
    source: 'Science',
    topic: 'Genetics',
    date: '2023-10-10',
    tags: ['gene therapy', 'rare disease', 'CRISPR'],
  },
];

// Define columns for the table
const columns: TableProps<DataType>['columns'] = [
  {
    title: 'Title',
    dataIndex: 'title',
    key: 'title',
    render: (text) => <a>{text}</a>,
  },
  {
    title: 'Source',
    dataIndex: 'source',
    key: 'source',
  },
  {
    title: 'Topic',
    dataIndex: 'topic',
    key: 'topic',
    filters: [
      { text: 'Oncology', value: 'Oncology' },
      { text: 'Neurology', value: 'Neurology' },
      { text: 'Infectious Disease', value: 'Infectious Disease' },
      { text: 'Genetics', value: 'Genetics' },
    ],
    onFilter: (value, record) => record.topic.indexOf(value as string) === 0,
  },
  {
    title: 'Date',
    dataIndex: 'date',
    key: 'date',
    sorter: (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
  },
  {
    title: 'Tags',
    key: 'tags',
    dataIndex: 'tags',
    render: (_, { tags }) => (
      <>
        {tags.map((tag) => {
          let color = tag.length > 5 ? 'geekblue' : 'green';
          if (tag === 'FDA') {
            color = 'volcano';
          }
          return (
            <Tag color={color} key={tag}>
              {tag.toUpperCase()}
            </Tag>
          );
        })}
      </>
    ),
  },
  {
    title: 'Action',
    key: 'action',
    render: () => (
      <Space size="middle">
        <a>Read</a>
        <a>Save</a>
      </Space>
    ),
  },
];

// Statistics data
const stats = [
  { title: 'Articles', value: 2458 },
  { title: 'Sources', value: 42 },
  { title: 'Topics', value: 18 },
  { title: 'Updates Today', value: 124 },
];

export default function AntDesignExample() {
  const [searchText, setSearchText] = useState('');
  const [selectedTopic, setSelectedTopic] = useState<string>('all');
  const [activeTab, setActiveTab] = useState('1');

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <Title level={2}>BioPharma Insights with Ant Design</Title>
      <Paragraph className="mb-6">
        This example demonstrates how Ant Design components can be integrated with your Next.js application.
      </Paragraph>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {stats.map((stat, index) => (
          <Card key={index}>
            <Statistic title={stat.title} value={stat.value} />
          </Card>
        ))}
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <Input
          placeholder="Search articles..."
          prefix={<SearchOutlined />}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          style={{ width: '100%' }}
        />
        <Space>
          <Select
            defaultValue="all"
            style={{ width: 200 }}
            onChange={(value) => setSelectedTopic(value)}
          >
            <Option value="all">All Topics</Option>
            <Option value="oncology">Oncology</Option>
            <Option value="neurology">Neurology</Option>
            <Option value="infectious">Infectious Disease</Option>
            <Option value="genetics">Genetics</Option>
          </Select>
          <DatePicker placeholder="Filter by date" />
          <Button type="primary" icon={<FilterOutlined />}>
            Apply Filters
          </Button>
        </Space>
      </div>

      {/* Tabs for different views */}
      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab} 
        className="mb-6"
        items={[
          {
            key: '1',
            label: (
              <span>
                <FileTextOutlined />
                Articles
              </span>
            ),
            children: <Table columns={columns} dataSource={data} />
          },
          {
            key: '2',
            label: (
              <span>
                <BarChartOutlined />
                Analytics
              </span>
            ),
            children: (
              <div className="bg-gray-100 p-8 rounded-lg text-center">
                <p>Analytics charts would appear here</p>
              </div>
            )
          },
          {
            key: '3',
            label: (
              <span>
                <BellOutlined />
                Notifications
              </span>
            ),
            children: (
              <div className="bg-gray-100 p-8 rounded-lg text-center">
                <p>Notification settings would appear here</p>
              </div>
            )
          }
        ]}
      />

      {/* Action buttons */}
      <div className="flex justify-end space-x-4">
        <Button>Cancel</Button>
        <Button type="primary">Save Changes</Button>
      </div>
    </div>
  );
} 