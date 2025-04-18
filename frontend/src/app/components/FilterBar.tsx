'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { 
  Input, 
  Card, 
  Tag, 
  Space, 
  Typography, 
  Select,
  Row,
  Col,
  Grid
} from 'antd';
import { 
  SearchOutlined, 
  ArrowUpOutlined,
  ArrowDownOutlined,
  CalendarOutlined,
  FontSizeOutlined,
  GlobalOutlined,
  TagsOutlined
} from '@ant-design/icons';
import { TopicStat } from '@/types'; // Import TopicStat type
import debounce from 'lodash.debounce'; // Need to install lodash.debounce

const { Text } = Typography;
const { Search } = Input;
const { Option } = Select;
const { useBreakpoint } = Grid;

// Define available sort options
const sortOptions = [
  { value: 'pub_date', label: 'Date', icon: <CalendarOutlined /> },
  { value: 'title', label: 'Title', icon: <FontSizeOutlined /> },
  { value: 'source', label: 'Source', icon: <GlobalOutlined /> },
  { value: 'topic', label: 'Topic', icon: <TagsOutlined /> },
];

interface FilterBarProps {
  // Receive current filter/sort values from parent Server Component
  currentTopic: string;
  currentSearch: string;
  currentSortBy: string;
  currentSortOrder: 'asc' | 'desc';
  topics: TopicStat[]; // Receive topics fetched by parent
  onTopicChange: (topic: string) => void;
  onSortChange: (value: string) => void;
}

export default function FilterBar({
  currentTopic = 'All',
  currentSearch = '',
  currentSortBy = 'pub_date',
  currentSortOrder = 'desc',
  topics = [],
  onTopicChange,
  onSortChange
}: FilterBarProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const screens = useBreakpoint();

  // Use local state for controlled search input to allow debouncing
  const [localSearch, setLocalSearch] = useState(currentSearch);

  // Update local search if the URL search param changes externally
  useEffect(() => {
    setLocalSearch(currentSearch);
  }, [currentSearch]);

  // Function to update URL query parameters
  const updateQueryParams = useCallback((newParams: Record<string, string>) => {
    const current = new URLSearchParams(Array.from(searchParams.entries()));

    // Update or remove parameters
    Object.entries(newParams).forEach(([key, value]) => {
      if (value) {
        current.set(key, value);
      } else {
        current.delete(key);
      }
    });

    // Reset page to 1 when filters/sort change
    current.set('page', '1');

    const query = current.toString();
    // Use router.push for navigation, triggering Server Component refetch
    router.push(`${pathname}?${query}`);
  }, [searchParams, router, pathname]);

  // Debounced function for search updates
  const debouncedSearchUpdate = useCallback(
    debounce((value: string) => {
      updateQueryParams({ search: value });
    }, 500), // 500ms debounce delay
    [updateQueryParams]
  );

  // Handle search input change
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setLocalSearch(value);
    debouncedSearchUpdate(value);
  };

  // Handle topic selection
  const handleTopicChange = (topic: string) => {
    onTopicChange(topic === 'All' ? '' : topic);
  };

  // Handle sort field change
  const handleSortByChange = (value: string) => {
    onSortChange(value);
  };

  // Handle sort order change
  const handleSortOrderChange = (value: 'asc' | 'desc') => {
    updateQueryParams({ sort_order: value });
  };

  // Determine layout based on screen size
  const isMobile = !screens.md;

  return (
    <Card style={{ marginBottom: 24 }}>
      <Row gutter={[16, 16]} align="middle">
        {/* Search Input */}
        <Col xs={24} md={10} lg={8}>
          <Search
            placeholder="Search articles..."
            value={localSearch} // Use local state for input value
            onChange={handleSearchChange} // Use debounced handler
            allowClear
            prefix={<SearchOutlined />}
          />
        </Col>

        {/* Sort Controls */}
        <Col xs={24} sm={12} md={7} lg={6}>
          <Select
            value={currentSortBy}
            onChange={handleSortByChange}
            style={{ width: '100%' }}
            aria-label="Sort by"
          >
            {sortOptions.map(option => (
              <Option key={option.value} value={option.value}>
                <Space size="small">
                  {option.icon}
                  {option.label}
                </Space>
              </Option>
            ))}
          </Select>
        </Col>
        <Col xs={24} sm={12} md={3} lg={2}>
          <Select
            value={currentSortOrder}
            onChange={handleSortOrderChange}
            style={{ width: '100%' }}
            aria-label="Sort order"
          >
            <Option value="desc">
              <Space size="small"><ArrowDownOutlined /> Desc</Space>
            </Option>
            <Option value="asc">
              <Space size="small"><ArrowUpOutlined /> Asc</Space>
            </Option>
          </Select>
        </Col>

        {/* Topic Filter Tags (shown below on mobile) */}
        {!isMobile && (
           <Col md={24} lg={8} style={{ textAlign: 'right' }}>
             <Space wrap size={[8, 8]} style={{ justifyContent: 'flex-end'}}>
               <Tag
                  key="All"
                  color={currentTopic === 'All' ? 'blue' : 'default'}
                  style={{ cursor: 'pointer', margin: '2px' }}
                  onClick={() => handleTopicChange('All')}
                >
                  All Topics
                </Tag>
                {topics.map((topicStat) => (
                  <Tag
                    key={topicStat.topic}
                    color={currentTopic === topicStat.topic ? 'blue' : 'default'}
                    style={{ cursor: 'pointer', margin: '2px' }}
                    onClick={() => handleTopicChange(topicStat.topic)}
                  >
                    {topicStat.topic} ({topicStat.count})
                  </Tag>
                ))}
              </Space>
           </Col>
        )}
      </Row>

      {/* Topic Filter Tags (Mobile Layout) */}
      {isMobile && (
        <Row gutter={[8, 8]} style={{ marginTop: 16 }}>
          <Col span={24}>
            <Space wrap size={[8, 8]}>
               <Tag
                  key="All-mobile"
                  color={currentTopic === 'All' ? 'blue' : 'default'}
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleTopicChange('All')}
                >
                  All Topics
                </Tag>
              {topics.map((topicStat) => (
                <Tag
                  key={`${topicStat.topic}-mobile`}
                  color={currentTopic === topicStat.topic ? 'blue' : 'default'}
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleTopicChange(topicStat.topic)}
                >
                  {topicStat.topic} ({topicStat.count})
                </Tag>
              ))}
            </Space>
          </Col>
        </Row>
      )}

      {/* Active filters display (Optional - might be redundant if controls reflect state) */}
      {/* ... could add a display for active filters if needed ... */}

    </Card>
  );
} 