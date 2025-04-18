import { 
  Space, 
  Select, 
  DatePicker, 
  Input, 
  Button, 
  Card,
  Tooltip,
  Tag
} from 'antd';
import { 
  SearchOutlined, 
  FilterOutlined, 
  SortAscendingOutlined,
  ClockCircleOutlined,
  FileTextOutlined
} from '@ant-design/icons';
import { useState } from 'react';
import { TOPICS } from '../../constants';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;
const { Option } = Select;

interface ArticleFiltersProps {
  onSearch: (query: string) => void;
  onTopicChange: (topic: string) => void;
  onSourceChange: (source: string) => void;
  onDateRangeChange: (dates: [Date | null, Date | null]) => void;
  onSortChange: (sort: { field: string; order: string }) => void;
  onReadingTimeChange: (time: string) => void;
  onHasFullContentChange: (hasFullContent: boolean) => void;
  selectedTopic?: string;
  selectedSource?: string;
  dateRange?: [Date | null, Date | null];
  sortBy?: string;
  sortOrder?: string;
  readingTime?: string;
  hasFullContent?: boolean;
  sources: string[];
  loading?: boolean;
}

export const ArticleFilters: React.FC<ArticleFiltersProps> = ({
  onSearch,
  onTopicChange,
  onSourceChange,
  onDateRangeChange,
  onSortChange,
  onReadingTimeChange,
  onHasFullContentChange,
  selectedTopic = 'All',
  selectedSource = 'All',
  dateRange,
  sortBy = 'pub_date',
  sortOrder = 'desc',
  readingTime,
  hasFullContent,
  sources,
  loading = false
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  // Handle search input
  const handleSearch = () => {
    onSearch(searchQuery);
  };

  // Handle search on enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <Card className="article-filters">
      <Space direction="vertical" size="middle" className="w-full">
        {/* Search and Topic Row */}
        <Space wrap>
          <Input
            placeholder="Search articles..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            prefix={<SearchOutlined />}
            style={{ width: 250 }}
            allowClear
          />
          
          <Select
            value={selectedTopic}
            onChange={onTopicChange}
            style={{ width: 200 }}
            placeholder="Select topic"
          >
            <Option value="All">All Topics</Option>
            {Object.keys(TOPICS).map((topic) => (
              <Option key={topic} value={topic}>
                {topic}
              </Option>
            ))}
          </Select>

          <Select
            value={selectedSource}
            onChange={onSourceChange}
            style={{ width: 200 }}
            placeholder="Select source"
          >
            <Option value="All">All Sources</Option>
            {sources.map((source) => (
              <Option key={source} value={source}>
                {source}
              </Option>
            ))}
          </Select>
        </Space>

        {/* Filters Row */}
        <Space wrap>
          <RangePicker
            onChange={(dates) => {
              onDateRangeChange([
                dates?.[0]?.toDate() || null,
                dates?.[1]?.toDate() || null
              ]);
            }}
            value={dateRange ? [
              dateRange[0] ? dayjs(dateRange[0]) : null,
              dateRange[1] ? dayjs(dateRange[1]) : null
            ] : null}
          />

          <Select
            value={`${sortBy}-${sortOrder}`}
            onChange={(value) => {
              const [field, order] = value.split('-');
              onSortChange({ field, order });
            }}
            style={{ width: 180 }}
          >
            <Option value="pub_date-desc">Newest First</Option>
            <Option value="pub_date-asc">Oldest First</Option>
            <Option value="title-asc">Title A-Z</Option>
            <Option value="title-desc">Title Z-A</Option>
            <Option value="reading_time-asc">Reading Time ↑</Option>
            <Option value="reading_time-desc">Reading Time ↓</Option>
          </Select>

          <Select
            value={readingTime}
            onChange={onReadingTimeChange}
            style={{ width: 160 }}
            placeholder="Reading time"
          >
            <Option value="">Any length</Option>
            <Option value="short">Short (&lt; 5 min)</Option>
            <Option value="medium">Medium (5-10 min)</Option>
            <Option value="long">Long (&gt; 10 min)</Option>
          </Select>

          <Tooltip title="Show only articles with full content">
            <Button
              type={hasFullContent ? 'primary' : 'default'}
              icon={<FileTextOutlined />}
              onClick={() => onHasFullContentChange(!hasFullContent)}
            >
              Full Content
            </Button>
          </Tooltip>
        </Space>

        {/* Active Filters */}
        <Space wrap>
          {selectedTopic !== 'All' && (
            <Tag closable onClose={() => onTopicChange('All')}>
              Topic: {selectedTopic}
            </Tag>
          )}
          {selectedSource !== 'All' && (
            <Tag closable onClose={() => onSourceChange('All')}>
              Source: {selectedSource}
            </Tag>
          )}
          {dateRange?.[0] && dateRange?.[1] && (
            <Tag closable onClose={() => onDateRangeChange([null, null])}>
              Date: {dayjs(dateRange[0]).format('MMM D')} - {dayjs(dateRange[1]).format('MMM D')}
            </Tag>
          )}
          {readingTime && (
            <Tag closable onClose={() => onReadingTimeChange('')}>
              Reading Time: {readingTime}
            </Tag>
          )}
          {hasFullContent && (
            <Tag closable onClose={() => onHasFullContentChange(false)}>
              Full Content Only
            </Tag>
          )}
        </Space>
      </Space>
    </Card>
  );
};

// Add styles to your global CSS or a separate module
const styles = `
.article-filters {
  margin-bottom: 24px;
}

.article-filters .ant-space {
  width: 100%;
}

.article-filters .ant-input-affix-wrapper {
  border-radius: 4px;
}

.article-filters .ant-select {
  min-width: 120px;
}

.article-filters .ant-tag {
  margin: 4px;
  padding: 4px 8px;
  border-radius: 4px;
}

@media (max-width: 576px) {
  .article-filters .ant-space {
    gap: 8px !important;
  }
  
  .article-filters .ant-input-affix-wrapper,
  .article-filters .ant-select,
  .article-filters .ant-picker {
    width: 100% !important;
  }
}
`; 