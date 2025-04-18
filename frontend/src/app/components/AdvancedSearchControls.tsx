'use client';

import { useState } from 'react';
import { Row, Col, Typography, Select, Slider } from 'antd';

const { Text } = Typography;
const { Option } = Select;

// Search options interface
export interface SearchOptions {
  searchType?: 'auto' | 'exact' | 'fuzzy' | 'semantic';
  threshold?: number;
  fields?: string[];
}

interface AdvancedSearchControlsProps {
  visible: boolean;
  onSearch: (value: string, options: SearchOptions) => void;
  currentSearchValue?: string;
}

export const AdvancedSearchControls = ({ 
  visible,
  onSearch,
  currentSearchValue = ''
}: AdvancedSearchControlsProps) => {
  const [searchType, setSearchType] = useState<'auto' | 'exact' | 'fuzzy' | 'semantic'>('auto');
  const [threshold, setThreshold] = useState(0.6);
  const [selectedFields, setSelectedFields] = useState<string[]>(['title', 'topic', 'description']);

  const handleAdvancedSearch = () => {
    onSearch(currentSearchValue, {
      searchType,
      threshold,
      fields: selectedFields
    });
  };

  // Apply search options when they change
  const handleOptionChange = () => {
    if (currentSearchValue) {
      handleAdvancedSearch();
    }
  };

  if (!visible) return null;

  return (
    <div className="advanced-search-controls" style={{ marginTop: 16 }}>
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={6}>
          <div>
            <Text strong>Search Type</Text>
            <Select
              style={{ width: '100%', marginTop: 4 }}
              value={searchType}
              onChange={(value) => {
                setSearchType(value);
                setTimeout(handleOptionChange, 100);
              }}
            >
              <Option value="auto">Automatic</Option>
              <Option value="exact">Exact Match</Option>
              <Option value="fuzzy">Fuzzy Search</Option>
              <Option value="semantic">Semantic Search</Option>
            </Select>
          </div>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <div>
            <Text strong>Match Threshold</Text>
            <Slider
              min={0}
              max={1}
              step={0.1}
              value={threshold}
              onChange={(value) => {
                setThreshold(value);
                setTimeout(handleOptionChange, 100);
              }}
              disabled={searchType === 'exact'}
              marks={{
                0: '0',
                0.5: '0.5',
                1: '1'
              }}
            />
          </div>
        </Col>

        <Col xs={24} md={12}>
          <div>
            <Text strong>Search Fields</Text>
            <Select
              mode="multiple"
              style={{ width: '100%', marginTop: 4 }}
              value={selectedFields}
              onChange={(value) => {
                setSelectedFields(value);
                setTimeout(handleOptionChange, 100);
              }}
              options={[
                { label: 'Title', value: 'title' },
                { label: 'Topic', value: 'topic' },
                { label: 'Description', value: 'description' },
                { label: 'Source', value: 'source' },
                { label: 'Content', value: 'document' }
              ]}
            />
          </div>
        </Col>
      </Row>

      <div style={{ marginTop: 8 }}>
        <Text type="secondary" style={{ fontSize: '12px' }}>
          Search options will be applied automatically when changed.
        </Text>
      </div>
    </div>
  );
};

export default AdvancedSearchControls; 