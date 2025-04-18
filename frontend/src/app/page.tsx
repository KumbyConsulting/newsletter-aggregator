'use client';
// This is now a Client Component since it uses Ant Design UI components

import { useEffect, useState, useMemo, useCallback, useRef } from 'react';
import {
  Layout,
  Typography,
  Row,
  Col,
  Card,
  Space,
  Spin,
  Input,
  Button,
  Select,
  Divider,
  Tag,
  Slider,
  Skeleton,
} from 'antd';
import {
  ClockCircleOutlined,
  SearchOutlined,
  SyncOutlined,
  FilterOutlined,
  SortAscendingOutlined,
  SortDescendingOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import debounce from 'lodash/debounce';
import dynamic from 'next/dynamic';

// Import API functions and types
import {
  getArticles as fetchArticlesService, 
  getTopicStats, 
  Article as ApiArticle,
  mapCoreToArticle
} from '@/app/services/api';
import { Article, ArticlesApiResponse, TopicStat, TopicsApiResponse, ApiErrorResponse } from '@/types';

// Import placeholder components (we'll create/update these next)
import FilterBar from './components/FilterBar';
import { ArticlesGrid } from './components/ArticlesGrid';
import PaginationControls from './components/PaginationControls';
import TopicDistribution from './components/TopicDistribution';
import ErrorDisplay from './components/ErrorDisplay';
import ClientInteractions from './components/ClientInteractions';
import { useSearchParams, useRouter } from 'next/navigation';
import { SearchOptions } from './components/AdvancedSearchControls';

const { Content } = Layout;
const { Title, Text, Paragraph } = Typography;
const { Search } = Input;
const { Option } = Select;

// Dynamically import Advanced Search Controls
const DynamicAdvancedSearchControls = dynamic(
  () => import('./components/AdvancedSearchControls'),
  { 
    ssr: false,
    loading: () => <Skeleton active paragraph={{ rows: 3 }} />
  }
);

// Helper to safely parse numbers from search params
const safeParseInt = (value: string | undefined, defaultValue: number): number => {
  if (value === undefined) return defaultValue;
  const parsed = parseInt(value, 10);
  return isNaN(parsed) ? defaultValue : parsed;
};

// Helper to validate sort order
const getValidSortOrder = (order: string | undefined): 'asc' | 'desc' => {
  if (order === 'asc' || order === 'desc') {
    return order;
  }
  return 'desc'; // Default sort order
};

// Cache interface
interface CacheEntry {
  articles: Article[];
  pagination: {
    page: number;
    total: number;
    total_pages: number;
  };
  topics: TopicStat[];
  timestamp: number;
}

// Main page component - Now a Client Component
export default function Home() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [articles, setArticles] = useState<Article[]>([]);
  const [topics, setTopics] = useState<TopicStat[]>([]);
  const [pagination, setPagination] = useState({ page: 1, total: 0, total_pages: 1 });
  const [isLoading, setIsLoading] = useState(true);
  const [fetchError, setFetchError] = useState<ApiErrorResponse | null>(null);
  const [searchValue, setSearchValue] = useState('');
  const [advancedSearchVisible, setAdvancedSearchVisible] = useState(false);
  const [searchCache, setSearchCache] = useState<Record<string, CacheEntry>>({});

  // Get search parameters
  const currentPage = safeParseInt(searchParams.get('page') || '1', 1);
  const currentLimit = safeParseInt(searchParams.get('limit') || '10', 10);
  const currentTopic = searchParams.get('topic') || 'All';
  const currentSearch = searchParams.get('search') || '';
  const currentSortBy = searchParams.get('sort_by') || 'pub_date';
  const currentSortOrder = getValidSortOrder(searchParams.get('sort_order') || 'desc');

  // Initialize search value from URL on mount
  useEffect(() => {
    setSearchValue(currentSearch);
  }, [currentSearch]);

  // Cache expiration check
  const isCacheValid = useCallback((cacheKey: string): boolean => {
    if (!searchCache[cacheKey]) return false;
    
    // Cache expires after 2 minutes
    const now = Date.now();
    const cacheAge = now - searchCache[cacheKey].timestamp;
    return cacheAge < 120000; // 2 minutes in milliseconds
  }, [searchCache]);

  // Modified fetch data with caching to support single page loading
  const fetchDataWithCache = useCallback(async (options: { page?: number } = {}) => {
    const { page = currentPage } = options;
    
    // --- Start Read Advanced Params from URL ---
    const searchType = searchParams.get('searchType') || undefined;
    const thresholdParam = searchParams.get('threshold');
    const threshold = thresholdParam ? parseFloat(thresholdParam) : undefined;
    const fieldsParam = searchParams.get('fields');
    const fields = fieldsParam ? fieldsParam.split(',') : undefined;
    // --- End Read Advanced Params from URL ---
    
    const params = {
      page,
      limit: currentLimit,
      topic: currentTopic,
      search: currentSearch,
      sort_by: currentSortBy,
      sort_order: currentSortOrder,
      // --- Start Add Advanced Params to request object ---
      searchType,
      threshold,
      fields
      // --- End Add Advanced Params to request object ---
    };
    
    // Create a cache key from the search parameters
    const cacheKey = JSON.stringify(params);
    console.log(`Fetching data with cache key: ${cacheKey}`);
    
    // Check if we have a valid cached result
    if (isCacheValid(cacheKey)) {
      const cached = searchCache[cacheKey];
      setArticles(cached.articles);
      setTopics(cached.topics);
      setPagination({
        page: cached.pagination.page,
        total: cached.pagination.total,
        total_pages: Math.max(1, cached.pagination.total_pages || 1)
      });
      setFetchError(null);
      return Promise.resolve();
    }
    
    // Not in cache or cache expired, fetch from API
    setIsLoading(true);
    
    try {
      // Get articles from service API - pass advanced params
      const articlesData = await fetchArticlesService(
        params.page, 
        params.limit, 
        params.search, 
        params.topic !== 'All' ? params.topic : '',
        params.sort_by, 
        params.sort_order,
        '', // source - not used here
        '', // dateFrom - not used here
        '', // dateTo - not used here
        '', // readingTime - not used here
        false, // hasSummary - not used here
        false, // hasFullContent - not used here
        // --- Pass Advanced Params to API Call ---
        params.searchType,
        params.threshold,
        params.fields
        // --- End Pass Advanced Params to API Call ---
      );
      
      // Get topics stats
      const topicsData = await getTopicStats();
      
      // Make sure we have valid data before updating state
      if (articlesData && articlesData.articles) {
        // Calculate total_pages if not provided or invalid
        const total_pages = articlesData.total_pages || 
                          Math.max(1, Math.ceil((articlesData.total || articlesData.articles.length) / params.limit));
        
        console.log('API response pagination:', {
          page: articlesData.page,
          total: articlesData.total,
          total_pages: articlesData.total_pages,
          calculated_total_pages: total_pages,
          articles_count: articlesData.articles.length
        });
        
        // Cache the results
        const cacheEntry = {
          articles: articlesData.articles,
          topics: topicsData.topics,
          pagination: {
            page: articlesData.page,
            total: articlesData.total || articlesData.articles.length,
            total_pages: total_pages
          },
          timestamp: Date.now()
        };
        
        setSearchCache(prev => ({
          ...prev,
          [cacheKey]: cacheEntry
        }));
        
        // Update state with new data
        setArticles(articlesData.articles);
        setTopics(topicsData.topics);
        setPagination({
          page: articlesData.page,
          total: articlesData.total || articlesData.articles.length,
          total_pages: total_pages
        });
        setFetchError(null);
      } else {
        throw new Error('Invalid API response format');
      }
    } catch (error: any) {
      console.error("Error fetching data for Home page:", error);
      if (typeof error === 'object' && error !== null && 'error' in error && 'status_code' in error) {
        setFetchError(error as ApiErrorResponse);
      } else {
        setFetchError({
          error: 'An unexpected error occurred while loading data.',
          status_code: 500
        });
      }
      
      // Re-throw the error so the calling code can handle it
      throw error;
    } finally {
      // Always reset loading state regardless of success or failure
      setIsLoading(false);
    }
  }, [currentPage, currentLimit, currentTopic, currentSearch, currentSortBy, currentSortOrder, searchParams, isCacheValid, searchCache]);

  // Initial data fetch on mount or when search criteria change
  useEffect(() => {
    console.log('Search criteria changed, fetching page:', currentPage);
    fetchDataWithCache({ page: currentPage })
      .catch(error => {
        console.error('Failed to fetch articles:', error);
        // Loading state is reset in the finally block of fetchDataWithCache
      });
  }, [currentTopic, currentSearch, currentSortBy, currentSortOrder, currentLimit, currentPage, fetchDataWithCache]);

  // Debounced search handler
  const debouncedSearch = useMemo(
    () => debounce((value: string, options: SearchOptions = {}) => {
      const params = new URLSearchParams(searchParams.toString());
      if (value) {
        params.set('search', value);
        // Add search type if specified
        if (options.searchType && options.searchType !== 'auto') {
          params.set('searchType', options.searchType);
        } else {
          params.delete('searchType');
        }
        // Add threshold if specified
        if (options.threshold) {
          params.set('threshold', options.threshold.toString());
        }
        // Add fields if specified
        if (options.fields?.length) {
          params.set('fields', options.fields.join(','));
        }
      } else {
        params.delete('search');
        params.delete('searchType');
        params.delete('threshold');
        params.delete('fields');
      }
      params.set('page', '1'); // Reset to first page on new search
      router.push(`/?${params.toString()}`);
    }, 300),
    [searchParams, router]
  );

  // Update handleSearch to support enhanced search and use debounced version
  const handleSearch = useCallback((value: string, options: SearchOptions = {}) => {
    setSearchValue(value);
    debouncedSearch(value, options);
  }, [debouncedSearch]);

  // Handle topic change
  const handleTopicChange = useCallback((value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (value && value !== 'All') {
      params.set('topic', value);
    } else {
      params.delete('topic');
    }
    params.set('page', '1'); // Reset to first page
    router.push(`/?${params.toString()}`);
  }, [searchParams, router]);

  // Handle sort change
  const handleSortChange = useCallback((value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('sort_by', value);
    router.push(`/?${params.toString()}`);
  }, [searchParams, router]);

  // Toggle sort order
  const toggleSortOrder = useCallback(() => {
    const params = new URLSearchParams(searchParams.toString());
    const newOrder = currentSortOrder === 'asc' ? 'desc' : 'asc';
    params.set('sort_order', newOrder);
    router.push(`/?${params.toString()}`);
  }, [currentSortOrder, searchParams, router]);

  // Reset all filters
  const resetFilters = useCallback(() => {
    // Clear cache when resetting filters
    setSearchCache({});
    router.push('/');
  }, [router]);
  
  // Toggle advanced search visibility
  const toggleAdvancedSearch = useCallback(() => {
    setAdvancedSearchVisible(!advancedSearchVisible);
  }, [advancedSearchVisible]);

  // Memoize components for better performance
  const memoizedArticlesGrid = useMemo(() => (
    <ArticlesGrid articles={articles} loading={isLoading} />
  ), [articles, isLoading]);

  const memoizedTopicDistribution = useMemo(() => (
    <TopicDistribution topics={topics} />
  ), [topics]);

  const memoizedPaginationControls = useMemo(() => {
    // Log pagination data for debugging
    console.log('Pagination data:', {
      page: pagination.page,
      total: pagination.total,
      total_pages: pagination.total_pages,
      currentLimit
    });
    
    // Ensure totalPages is at least 1 to always show pagination controls
    const totalPages = Math.max(1, pagination.total_pages);
    
    return (
      <div style={{ margin: '32px 0' }}>
        <PaginationControls
          currentPage={pagination.page}
          totalPages={totalPages}
          totalItems={pagination.total}
          itemsPerPage={currentLimit}
          disabled={isLoading}
        />
      </div>
    );
  }, [pagination, currentLimit, isLoading]);

  return (
    <Layout className="newsletter-layout">
      {/* ClientInteractions handles update button, status polling etc. */}
      <ClientInteractions />

      <Content className="main-content">
        <div className="container" style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 16px' }}>
          {/* Header Card */}
          <Card className="header-card mb-6">
            <Row justify="space-between" align="middle" gutter={[16, 16]}>
              <Col xs={24} md={16}>
                <Title level={2} style={{ marginBottom: 8, color: 'var(--primary-color, #00405e)' }}>
                  Kumby Consulting Newsboard
                </Title>
                <Paragraph className="tagline" style={{ color: 'var(--secondary-color, #7f9360)' }}>
                  Stay updated with the latest industry news, research, and regulatory developments across various business sectors.
                </Paragraph>
              </Col>
              {/* Update status shown in ClientInteractions */}
            </Row>
          </Card>

          {/* Search Form */}
          <Card className="search-form mb-6">
            <div className="search-form-header mb-4 flex justify-between items-center">
              <div>
                <Title level={4} style={{ marginBottom: 4 }}>Search Articles</Title>
                <Text type="secondary">Search through {pagination.total} articles</Text>
              </div>
              <Button 
                type="link" 
                icon={<FilterOutlined />} 
                onClick={toggleAdvancedSearch}
              >
                {advancedSearchVisible ? 'Simple Search' : 'Advanced Search'}
              </Button>
            </div>

            <Row gutter={16}>
              <Col xs={24} md={18} lg={20}>
                <Search
                  placeholder="Enter keywords, phrases, or article titles..."
                  allowClear
                  enterButton={<><SearchOutlined /> Search</>}
                  size="large"
                  value={searchValue}
                  onChange={(e) => setSearchValue(e.target.value)}
                  onSearch={(value) => handleSearch(value, {
                    searchType: searchParams.get('searchType') as any || 'auto',
                    threshold: parseFloat(searchParams.get('threshold') || '0.6'),
                    fields: searchParams.get('fields')?.split(',')
                  })}
                  loading={isLoading}
                />
                <div className="mt-1 text-xs text-gray-500">
                  {advancedSearchVisible ? (
                    'Use advanced options below to refine your search'
                  ) : (
                    'Use quotes for exact phrases, OR for alternatives, - to exclude terms'
                  )}
                </div>
              </Col>

              <Col xs={24} md={6} lg={4} style={{ display: 'flex', alignItems: 'flex-start' }}>
                <Button 
                  icon={<ReloadOutlined />} 
                  size="large" 
                  onClick={resetFilters}
                  style={{ width: '100%' }}
                >
                  Reset
                </Button>
              </Col>
            </Row>

            {/* Advanced Search Controls */}
            {advancedSearchVisible && (
              <DynamicAdvancedSearchControls 
                visible={advancedSearchVisible}
                onSearch={handleSearch}
                currentSearchValue={searchValue}
              />
            )}

            {/* Filter Controls */}
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              {/* Topic Filter */}
              <Col xs={24} sm={12} md={6}>
                <div>
                  <Text strong>Topic</Text>
                  <Select
                    style={{ width: '100%', marginTop: 4 }}
                    placeholder="Select a topic"
                    value={currentTopic === 'All' ? undefined : currentTopic}
                    onChange={handleTopicChange}
                    allowClear
                  >
                    <Option value="All">All Topics</Option>
                    {topics.map((topic) => (
                      <Option key={topic.topic} value={topic.topic}>
                        {topic.topic} ({topic.count})
                      </Option>
                    ))}
                  </Select>
                </div>
              </Col>

              {/* Sort Controls */}
              <Col xs={24} sm={12} md={6}>
                <div>
                  <Text strong>Sort By</Text>
                  <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                    <Select
                      style={{ flex: 1 }}
                      value={currentSortBy}
                      onChange={handleSortChange}
                    >
                      <Option value="pub_date">Date</Option>
                      <Option value="title">Title</Option>
                      <Option value="source">Source</Option>
                      <Option value="relevance">Relevance</Option>
                    </Select>
                    <Button 
                      icon={currentSortOrder === 'asc' ? <SortAscendingOutlined /> : <SortDescendingOutlined />}
                      onClick={toggleSortOrder}
                    />
                  </div>
                </div>
              </Col>
            </Row>

            {/* Active Filters */}
            {(currentTopic !== 'All' || currentSearch || searchParams.get('searchType')) && (
              <div style={{ marginTop: 16, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {currentSearch && (
                  <Tag 
                    className="filter-tag search-tag"
                    closable 
                    onClose={() => handleSearch('')}
                    style={{ display: 'flex', alignItems: 'center', padding: '4px 8px' }}
                  >
                    <SearchOutlined style={{ marginRight: 4 }} /> 
                    {currentSearch}
                    {searchParams.get('searchType') && (
                      <span className="ml-1 text-xs">
                        ({searchParams.get('searchType')})
                      </span>
                    )}
                  </Tag>
                )}
                {currentTopic !== 'All' && (
                  <Tag 
                    className="filter-tag topic-tag"
                    closable 
                    onClose={() => handleTopicChange('All')}
                    style={{ display: 'flex', alignItems: 'center', padding: '4px 8px' }}
                  >
                    <FilterOutlined style={{ marginRight: 4 }} /> {currentTopic}
                  </Tag>
                )}
              </div>
            )}
          </Card>

          {/* Topic Distribution */}
          <Card className="mb-6">
            <Title level={4}>Topic Distribution</Title>
            <Paragraph type="secondary" style={{ marginBottom: 16 }}>
              Distribution of articles across different topics
            </Paragraph>
            {isLoading ? (
              <div style={{ height: 200, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <Skeleton active paragraph={{ rows: 5 }} />
              </div>
            ) : (
              memoizedTopicDistribution
            )}
          </Card>

          {/* Display error if fetching failed */}
          {fetchError && (
            <ErrorDisplay 
              error={fetchError.error} 
              statusCode={fetchError.status_code} 
            />
          )}

          {/* Articles Section */}
          <div className="articles-section">
            <Title level={4}>
              {pagination.total} Articles Found
              {currentSearch && <span> for "{currentSearch}"</span>}
              {currentTopic !== 'All' && <span> in {currentTopic}</span>}
            </Title>
            
            {/* Skeleton Loading for Articles */}
            {isLoading ? (
              <div className="article-skeletons">
                <Row gutter={[16, 16]}>
                  {Array.from({ length: currentLimit }).map((_, index) => (
                    <Col xs={24} sm={12} md={8} key={index}>
                      <Card>
                        <Skeleton active avatar paragraph={{ rows: 3 }} />
                      </Card>
                    </Col>
                  ))}
                </Row>
              </div>
            ) : (
              memoizedArticlesGrid
            )}
          </div>

          {/* Pagination Controls - passes pagination data */}
          {memoizedPaginationControls}
        </div>
      </Content>
    </Layout>
  );
}
