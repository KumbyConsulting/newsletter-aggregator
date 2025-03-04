'use client';

import { useState, useEffect } from 'react';
import { 
  Input, 
  Card, 
  Tag, 
  Space, 
  Typography, 
  Button, 
  Skeleton, 
  Flex
} from 'antd';
import { 
  SearchOutlined, 
  CloseOutlined 
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Search } = Input;

interface TopicData {
  topic: string;
  count: number;
  percentage?: number;
}

interface FilterBarProps {
  onSearch: (query: string) => void;
  onTopicChange: (topic: string) => void;
  selectedTopic: string;
}

export default function FilterBar({ onSearch, onTopicChange, selectedTopic = 'All' }: FilterBarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [topics, setTopics] = useState<(TopicData | string)[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch available topics on component mount
  useEffect(() => {
    const fetchTopics = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('/api/topics');
        
        if (!response.ok) {
          throw new Error('Failed to fetch topics');
        }
        
        const data = await response.json();
        setTopics(['All', ...data.topics]);
      } catch (error) {
        console.error('Error fetching topics:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTopics();
  }, []);

  // Handle search submission
  const handleSearch = (value: string) => {
    onSearch(value);
  };

  // Handle topic selection
  const handleTopicChange = (topic: string) => {
    onTopicChange(topic);
  };

  // Helper function to get topic name
  const getTopicName = (topic: TopicData | string): string => {
    if (typeof topic === 'string') {
      return topic;
    }
    return topic.topic;
  };

  return (
    <Card style={{ marginBottom: 24 }}>
      <Flex vertical gap="middle">
        <Flex wrap="wrap" justify="space-between" align="center" gap={16}>
          {/* Search form */}
          <Search
            placeholder="Search articles..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onSearch={handleSearch}
            style={{ maxWidth: '100%', flex: 1 }}
            allowClear
          />

          {/* Topic filter */}
          <Space wrap>
            {isLoading ? (
              <Skeleton.Button active={true} size="small" block={false} shape="round" />
            ) : (
              topics.map((topic) => {
                const topicName = getTopicName(topic);
                return (
                  <Tag
                    key={topicName}
                    color={selectedTopic === topicName ? 'blue' : 'default'}
                    style={{ 
                      cursor: 'pointer', 
                      padding: '4px 12px',
                      fontSize: '14px' 
                    }}
                    onClick={() => handleTopicChange(topicName)}
                  >
                    {topicName}
                  </Tag>
                );
              })
            )}
          </Space>
        </Flex>

        {/* Active filters display */}
        {(selectedTopic !== 'All' || searchQuery) && (
          <Flex align="center">
            <Text type="secondary" style={{ marginRight: 8 }}>Active filters:</Text>
            <Space wrap>
              {selectedTopic !== 'All' && (
                <Tag 
                  color="blue"
                  closable
                  onClose={() => handleTopicChange('All')}
                >
                  Topic: {selectedTopic}
                </Tag>
              )}
              {searchQuery && (
                <Tag 
                  color="blue"
                  closable
                  onClose={() => {
                    setSearchQuery('');
                    onSearch('');
                  }}
                >
                  Search: {searchQuery}
                </Tag>
              )}
            </Space>
          </Flex>
        )}
      </Flex>
    </Card>
  );
} 