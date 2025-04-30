'use client';
// This is now a Client Component since it uses Ant Design UI components

import { useEffect, useState, useRef, Suspense } from 'react';
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
import { useRouter, useSearchParams as useNextSearchParams } from 'next/navigation';

// Import API functions and types
import {
  getArticles as fetchArticlesService, 
  getTopicStats, 
  Article as ApiArticle,
  mapCoreToArticle
} from '@/services/api';
import { Article, ArticlesApiResponse, TopicStat, TopicsApiResponse, ApiErrorResponse } from '@/types';

// Import placeholder components (we'll create/update these next)
import FilterBar from './components/FilterBar';
import { ArticlesGrid } from './components/ArticlesGrid';
import PaginationControls from './components/PaginationControls';
import TopicDistribution from './components/TopicDistribution';
import ErrorDisplay from './components/ErrorDisplay';
import ClientInteractions from './components/ClientInteractions';
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

// SearchParamsWrapper - This is a separate client component to handle useSearchParams
function SearchParamsWrapper({ children }: { children: (params: URLSearchParams) => React.ReactNode }) {
  const searchParams = useNextSearchParams();
  return <>{children(searchParams)}</>;
}

// Helper for stable, sorted stringification
function stableStringify(obj: any) {
  return JSON.stringify(
    Object.keys(obj).sort().reduce((acc, key) => {
      acc[key] = obj[key];
      return acc;
    }, {} as any)
  );
}

// Main page component - Now a Client Component
export default function Home() {
  const router = useRouter();
  const [articles, setArticles] = useState<Article[]>([]);
  const [topics, setTopics] = useState<TopicStat[]>([]);
  const [pagination, setPagination] = useState({ page: 1, total: 0, total_pages: 1 });
  const [isLoading, setIsLoading] = useState(true);
  const [fetchError, setFetchError] = useState<ApiErrorResponse | null>(null);
  const [searchValue, setSearchValue] = useState('');
  const [advancedSearchVisible, setAdvancedSearchVisible] = useState(false);
  const [searchCache, setSearchCache] = useState<Map<string, CacheEntry>>(new Map());
  const [searchParamsObj, setSearchParamsObj] = useState<URLSearchParams | null>(null);

  // --- Derive all search/filter/sort state from searchParamsObj ---
  const currentPage = safeParseInt(searchParamsObj?.get('page') || '1', 1);
  const currentLimit = safeParseInt(searchParamsObj?.get('limit') || '10', 10);
  const currentTopic = searchParamsObj?.get('topic') || 'All';
  const currentSearch = searchParamsObj?.get('search') || '';
  const currentSortBy = searchParamsObj?.get('sort_by') || 'pub_date';
  const currentSortOrder = getValidSortOrder(searchParamsObj?.get('sort_order') || 'desc');

  // --- Cache expiration check ---
  const isCacheValid = (cacheKey: string): boolean => {
    const entry = searchCache.get(cacheKey);
    if (!entry) return false;
    const now = Date.now();
    const cacheAge = now - entry.timestamp;
    return cacheAge < 120000; // 2 minutes
  };

  // --- Keep a ref to latest searchParamsObj for debounce ---
  const searchParamsRef = useRef<URLSearchParams | null>(searchParamsObj);
  useEffect(() => { searchParamsRef.current = searchParamsObj; }, [searchParamsObj]);

  // --- Update search value from URL on mount or change ---
  useEffect(() => {
    setSearchValue(currentSearch);
  }, [currentSearch]);

  // --- Fetch data with cache ---
  async function fetchDataWithCache(options: { page?: number } = {}) {
    if (!searchParamsObj) return;
    const { page = currentPage } = options;
    const searchType = searchParamsObj.get('searchType') || undefined;
    const thresholdParam = searchParamsObj.get('threshold');
    const threshold = thresholdParam ? parseFloat(thresholdParam) : undefined;
    const fieldsParam = searchParamsObj.get('fields');
    const fields = fieldsParam ? fieldsParam.split(',') : undefined;
    const params = {
      page,
      limit: currentLimit,
      topic: currentTopic,
      search: currentSearch,
      sort_by: currentSortBy,
      sort_order: currentSortOrder,
      searchType,
      threshold,
      fields
    };
    const cacheKey = stableStringify(params);
    if (isCacheValid(cacheKey)) {
      const cached = searchCache.get(cacheKey)!;
      setArticles(cached.articles);
      setTopics(cached.topics);
      setPagination({
        page: cached.pagination.page,
        total: cached.pagination.total,
        total_pages: Math.max(1, cached.pagination.total_pages || 1)
      });
      setFetchError(null);
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    try {
      const articlesData = await fetchArticlesService(
        params.page,
        params.limit,
        params.search,
        params.topic !== 'All' ? params.topic : '',
        params.sort_by,
        params.sort_order,
        '', '', '', '', false, false,
        params.searchType,
        params.threshold,
        params.fields
      );
      let topicsData = { topics: [] as TopicStat[] };
      try {
        topicsData = await getTopicStats();
      } catch (topicError) {
        // Non-critical
      }
      if (articlesData && articlesData.articles) {
        const total_pages = articlesData.total_pages || Math.max(1, Math.ceil((articlesData.total || articlesData.articles.length) / params.limit));
        const cacheEntry: CacheEntry = {
          articles: articlesData.articles,
          topics: topicsData.topics,
          pagination: {
            page: articlesData.page,
            total: articlesData.total || articlesData.articles.length,
            total_pages: total_pages
          },
          timestamp: Date.now()
        };
        setSearchCache(prev => {
          const newCache = new Map(prev);
          newCache.set(cacheKey, cacheEntry);
          // Prune cache to 20 entries
          if (newCache.size > 20) {
            const firstKey = newCache.keys().next().value;
            newCache.delete(firstKey);
          }
          return newCache;
        });
        setArticles(articlesData.articles);
        setTopics(topicsData.topics);
        setPagination({
          page: articlesData.page,
          total: articlesData.total || articlesData.articles.length,
          total_pages: total_pages
        });
        setFetchError(null);
      } else {
        setFetchError({ error: 'Invalid API response format', status_code: 500 });
        setArticles([]);
        setPagination({ page: 1, total: 0, total_pages: 1 });
      }
    } catch (error: any) {
      let errorMessage = 'An unexpected error occurred while loading data.';
      let statusCode = 500;
      if (error.message && error.message.includes('timed out')) {
        errorMessage = 'The server took too long to respond. Please try again later.';
        statusCode = 504;
      } else if (typeof error === 'object' && error !== null) {
        if ('apiError' in error && error.apiError) {
          errorMessage = error.apiError.error || error.apiError.message || errorMessage;
          statusCode = error.apiError.status_code || error.statusCode || statusCode;
        } else if ('error' in error) {
          errorMessage = error.error;
          statusCode = error.status_code || statusCode;
        } else if ('message' in error) {
          errorMessage = error.message;
          if (error.message.includes('failed with status 502')) statusCode = 502;
          else if (error.message.includes('failed with status 503')) statusCode = 503;
          else if (error.message.includes('failed with status 504')) statusCode = 504;
        }
      }
      setFetchError({ error: errorMessage, status_code: statusCode });
      setArticles([]);
      setPagination({ page: 1, total: 0, total_pages: 1 });
    } finally {
      setIsLoading(false);
    }
  }

  // --- Fetch on mount or when search criteria change ---
  useEffect(() => {
    if (searchParamsObj) {
      fetchDataWithCache({ page: currentPage });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTopic, currentSearch, currentSortBy, currentSortOrder, currentLimit, currentPage, searchParamsObj]);

  // --- Handlers ---
  function handleSearchInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    setSearchValue(e.target.value);
  }

  function handleSearch(value: string, options: SearchOptions = {}) {
    const params = new URLSearchParams((searchParamsRef.current || new URLSearchParams()).toString());
    if (value) {
      params.set('search', value);
      if (options.searchType && options.searchType !== 'auto') {
        params.set('searchType', options.searchType);
      } else {
        params.delete('searchType');
      }
      if (options.threshold) {
        params.set('threshold', options.threshold.toString());
      }
      if (options.fields?.length) {
        params.set('fields', options.fields.join(','));
      }
    } else {
      params.delete('search');
      params.delete('searchType');
      params.delete('threshold');
      params.delete('fields');
    }
    params.set('page', '1');
    router.push(`/?${params.toString()}`);
  }

  function handleTopicChange(value: string) {
    const params = new URLSearchParams((searchParamsRef.current || new URLSearchParams()).toString());
    if (value && value !== 'All') {
      params.set('topic', value);
    } else {
      params.delete('topic');
    }
    params.set('page', '1');
    router.push(`/?${params.toString()}`);
  }

  function handleSortChange(value: string) {
    const params = new URLSearchParams((searchParamsRef.current || new URLSearchParams()).toString());
    params.set('sort_by', value);
    router.push(`/?${params.toString()}`);
  }

  function toggleSortOrder() {
    const params = new URLSearchParams((searchParamsRef.current || new URLSearchParams()).toString());
    const newOrder = currentSortOrder === 'asc' ? 'desc' : 'asc';
    params.set('sort_order', newOrder);
    router.push(`/?${params.toString()}`);
  }

  function resetFilters() {
    setSearchCache(new Map());
    router.push('/');
  }

  function toggleAdvancedSearch() {
    setAdvancedSearchVisible(v => !v);
  }

  // --- Memoized components (only for expensive renders) ---
  // (Removed unnecessary useMemo for simple components)

  // --- Skeleton loading logic ---
  const skeletonCount = Math.max(currentLimit, 6);

  return (
    <Layout className="newsletter-layout">
      <ClientInteractions />
      <Suspense fallback={<div className="p-8 text-center"><Spin size="large" /></div>}>
        <SearchParamsWrapper>
          {(searchParams) => {
            useEffect(() => {
              setSearchParamsObj(searchParams);
            }, [searchParams]);
            return (
              <Content className="main-content">
                <div className="container" style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 16px' }}>
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
                    </Row>
                  </Card>
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
                          onChange={handleSearchInputChange}
                          onSearch={value => handleSearch(value, {
                            searchType: (searchParams.get('searchType') as any) || 'auto',
                            threshold: parseFloat((searchParams.get('threshold') ?? '0.6')),
                            fields: (searchParams.get('fields') ?? '').split(',').filter(Boolean)
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
                    {advancedSearchVisible && (
                      <DynamicAdvancedSearchControls 
                        visible={advancedSearchVisible}
                        onSearch={handleSearch}
                        currentSearchValue={searchValue}
                      />
                    )}
                    <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
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
                      <TopicDistribution topics={topics} />
                    )}
                  </Card>
                  {fetchError && (
                    <ErrorDisplay 
                      error={fetchError.error} 
                      statusCode={fetchError.status_code} 
                    />
                  )}
                  <div className="articles-section">
                    <Title level={4}>
                      {pagination.total} Articles Found
                      {currentSearch && <span> for "{currentSearch}"</span>}
                      {currentTopic !== 'All' && <span> in {currentTopic}</span>}
                    </Title>
                    {isLoading ? (
                      <div className="article-skeletons">
                        <Row gutter={[16, 16]}>
                          {Array.from({ length: skeletonCount }).map((_, index) => (
                            <Col xs={24} sm={12} md={8} key={index}>
                              <Card>
                                <Skeleton active avatar paragraph={{ rows: 3 }} />
                              </Card>
                            </Col>
                          ))}
                        </Row>
                      </div>
                    ) : (
                      <ArticlesGrid articles={articles} loading={isLoading} />
                    )}
                  </div>
                  <div style={{ margin: '32px 0' }}>
                    <PaginationControls
                      currentPage={pagination.page}
                      totalPages={Math.max(1, pagination.total_pages)}
                      totalItems={pagination.total}
                      itemsPerPage={currentLimit}
                      disabled={isLoading}
                    />
                  </div>
                </div>
              </Content>
            );
          }}
        </SearchParamsWrapper>
      </Suspense>
    </Layout>
  );
}
