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
  DashboardOutlined,
  FireOutlined,
  CalendarOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import debounce from 'lodash/debounce';
import dynamic from 'next/dynamic';
import { useRouter, useSearchParams as useNextSearchParams } from 'next/navigation';

// Import API functions and types
import {
  getArticles as fetchArticlesService, 
  getTopicStats, 
  Article as ApiArticle,
  mapCoreToArticle,
  getTLDR,
  getDailyTrends,
  getWeeklyTrends,
  getWeeklyRecap,
  WeeklyRecapData,
  TLDRData,
  getSources,
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
import TLDRCard from './components/TLDRCard';
import WeeklyRecap from './components/WeeklyRecap';

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

// Helper to validate TLDR data
function isValidTLDR(data: any): boolean {
  return (
    data &&
    typeof data.summary === 'string' && data.summary.trim() &&
    Array.isArray(data.highlights) &&
    Array.isArray(data.sources) &&
    data.highlights.every((h: any) => h.title && h.url) &&
    data.sources.every((s: any) => s.title && s.url)
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
  const [retryCount, setRetryCount] = useState(0);
  const MAX_RETRIES = 2;
  const [tldr, setTldr] = useState<TLDRData | null>(null);
  const [dailyTrends, setDailyTrends] = useState<string[]>([]);
  const [weeklyTrends, setWeeklyTrends] = useState<string[]>([]);
  const [weeklyRecap, setWeeklyRecap] = useState<WeeklyRecapData | null>(null);
  const [weeklyRecapError, setWeeklyRecapError] = useState<string | null>(null);
  const [sources, setSources] = useState<{ name: string; count: number }[]>([]);
  const [currentSource, setCurrentSource] = useState<string>('');

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
    let attempts = 0;
    let lastError = null;
    while (attempts <= MAX_RETRIES) {
      try {
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
        } catch (topicError) {}
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
            const firstKey = newCache.keys().next().value;
            if (newCache.size > 20 && firstKey !== undefined) {
              newCache.delete(firstKey);
            }
            newCache.set(cacheKey, cacheEntry);
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
        setIsLoading(false);
        return;
      } catch (error: any) {
        lastError = error;
        if (
          error.message?.includes('timed out') ||
          error.message?.includes('Network') ||
          error.message?.includes('Failed to fetch')
        ) {
          attempts++;
          if (attempts > MAX_RETRIES) break;
          await new Promise(res => setTimeout(res, 1000 * attempts));
          continue;
        }
        break;
      }
    }
    setFetchError(lastError);
    setIsLoading(false);
  }

  // --- Fetch on mount or when search criteria change ---
  useEffect(() => {
    if (searchParamsObj) {
      fetchDataWithCache({ page: currentPage });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTopic, currentSearch, currentSortBy, currentSortOrder, currentLimit, currentPage, searchParamsObj, retryCount]);

  useEffect(() => {
    getTLDR().then(setTldr);
    getDailyTrends().then(data => setDailyTrends(data.topics));
    getWeeklyTrends().then(data => setWeeklyTrends(data.topics));
    getWeeklyRecap()
      .then(data => {
        // Defensive: check for correct shape
        if (
          typeof data?.recapSummary === 'string' &&
          Array.isArray(data?.highlights) &&
          data.highlights.every(
            h => typeof h.title === 'string' && typeof h.url === 'string'
          )
        ) {
          setWeeklyRecap(data);
          setWeeklyRecapError(null);
        } else {
          setWeeklyRecap(null);
          setWeeklyRecapError('Invalid data format from server.');
          console.error('WeeklyRecap: Invalid data format', data);
        }
      })
      .catch(err => {
        setWeeklyRecap(null);
        setWeeklyRecapError(
          err?.message || 'Failed to fetch weekly recap. Please try again later.'
        );
        console.error('WeeklyRecap: Error fetching data', err);
      });
    // Fetch sources for the filter dropdown
    getSources().then(res => {
      if (res && res.sources) setSources(res.sources);
    });
  }, []);

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

  // Handler for retry button in ErrorDisplay
  function handleRetry() {
    setRetryCount(c => c + 1);
  }

  function handleSourceChange(value: string) {
    setCurrentSource(value);
    const params = new URLSearchParams((searchParamsRef.current || new URLSearchParams()).toString());
    if (value) {
      params.set('source', value);
    } else {
      params.delete('source');
    }
    params.set('page', '1');
    router.push(`/?${params.toString()}`);
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
              setCurrentSource(searchParams.get('source') || '');
            }, [searchParams]);

            return (
              <Content className="main-content">
                <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
                  {/* Summary Section */}
                  <Row gutter={[24, 24]} className="mb-8">
                    <Col xs={24} lg={16}>
                      <Card className="h-full">
                        <div className="flex items-center mb-4">
                          <DashboardOutlined className="text-2xl text-blue-500 mr-2" />
                          <Title level={4} className="m-0">TL;DR</Title>
                        </div>
                        {tldr && isValidTLDR(tldr) ? (
                          <TLDRCard summary={tldr.summary} highlights={tldr.highlights} sources={tldr.sources} />
                        ) : (
                          <div className="text-gray-400 italic min-h-[40px]">
                            No summary available today. The news gods are silent.
                          </div>
                        )}
                      </Card>
                    </Col>
                    <Col xs={24} lg={8}>
                      <Card className="h-full">
                        <div className="flex items-center mb-4">
                          <CalendarOutlined className="text-2xl text-green-500 mr-2" />
                          <Title level={4} className="m-0">Weekly Recap</Title>
                        </div>
                        {weeklyRecapError ? (
                          <div className="text-red-500 min-h-[60px]">{weeklyRecapError}</div>
                        ) : weeklyRecap ? (
                          <WeeklyRecap
                            recapSummary={weeklyRecap.recapSummary}
                            highlights={weeklyRecap.highlights}
                          />
                        ) : (
                          <div className="min-h-[60px]">Loading...</div>
                        )}
                      </Card>
                    </Col>
                  </Row>

                  {/* Search Section */}
                  <Card className="mb-8">
                    <div className="flex justify-between items-center mb-6">
                      <div>
                        <Title level={4} className="mb-1">Search Articles</Title>
                        <Text type="secondary">Search through {pagination.total} articles</Text>
                      </div>
                      <Button 
                        type="link" 
                        icon={<FilterOutlined />} 
                        onClick={toggleAdvancedSearch}
                        className="flex items-center"
                      >
                        {advancedSearchVisible ? 'Simple Search' : 'Advanced Search'}
                      </Button>
                    </div>

                    <Row gutter={[16, 16]}>
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
                          className="w-full"
                        />
                        <div className="mt-2 text-xs text-gray-500">
                          {advancedSearchVisible 
                            ? 'Use advanced options below to refine your search'
                            : 'Use quotes for exact phrases, OR for alternatives, - to exclude terms'
                          }
                        </div>
                      </Col>
                      <Col xs={24} md={6} lg={4}>
                        <Button 
                          icon={<ReloadOutlined />} 
                          size="large" 
                          onClick={resetFilters}
                          className="w-full"
                        >
                          Reset
                        </Button>
                      </Col>
                    </Row>

                    {advancedSearchVisible && (
                      <div className="mt-4">
                        <DynamicAdvancedSearchControls 
                          visible={advancedSearchVisible}
                          onSearch={handleSearch}
                          currentSearchValue={searchValue}
                        />
                      </div>
                    )}

                    <Row gutter={[16, 16]} className="mt-6">
                      <Col xs={24} sm={12} md={8}>
                        <div className="space-y-1">
                          <Text strong>Topic</Text>
                          <Select
                            className="w-full"
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
                      <Col xs={24} sm={12} md={8}>
                        <div className="space-y-1">
                          <Text strong>Source</Text>
                          <Select
                            className="w-full"
                            placeholder="Select a source"
                            value={currentSource || undefined}
                            onChange={handleSourceChange}
                            allowClear
                            showSearch
                            optionFilterProp="children"
                          >
                            <Option value="">All Sources</Option>
                            {sources.map((source) => (
                              <Option key={source.name} value={source.name}>
                                {source.name} ({source.count})
                              </Option>
                            ))}
                          </Select>
                        </div>
                      </Col>
                      <Col xs={24} sm={12} md={8}>
                        <div className="space-y-1">
                          <Text strong>Sort By</Text>
                          <div className="flex gap-2">
                            <Select
                              className="flex-1"
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
                    {(currentTopic !== 'All' || currentSource || currentSearch || searchParams.get('searchType')) && (
                      <div className="flex flex-wrap gap-2 mt-4">
                        {currentSearch && (
                          <Tag 
                            className="filter-tag search-tag flex items-center px-2 py-1"
                            closable 
                            onClose={() => handleSearch('')}
                          >
                            <SearchOutlined className="mr-1" /> 
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
                            className="filter-tag topic-tag flex items-center px-2 py-1"
                            closable 
                            onClose={() => handleTopicChange('All')}
                          >
                            <FilterOutlined className="mr-1" /> {currentTopic}
                          </Tag>
                        )}
                        {currentSource && (
                          <Tag
                            className="filter-tag source-tag flex items-center px-2 py-1"
                            closable
                            onClose={() => handleSourceChange('')}
                          >
                            <FileTextOutlined className="mr-1" /> {currentSource}
                          </Tag>
                        )}
                      </div>
                    )}
                  </Card>

                  {/* Topic Distribution */}
                  <Card className="mb-8">
                    <Title level={4}>Topic Distribution</Title>
                    <Paragraph type="secondary" className="mb-4">
                      Distribution of articles across different topics
                    </Paragraph>
                    {isLoading ? (
                      <div className="h-[200px] flex justify-center items-center">
                        <Skeleton active paragraph={{ rows: 5 }} />
                      </div>
                    ) : (
                      <TopicDistribution topics={topics} />
                    )}
                  </Card>

                  {/* Article List */}
                  {fetchError && (
                    <ErrorDisplay 
                      error={fetchError.error} 
                      statusCode={fetchError.status_code} 
                      onRetry={handleRetry}
                    />
                  )}
                  
                  <div className="articles-section">
                    <Title level={4} className="mb-4">
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
                      <ArticlesGrid
                        articles={articles}
                        loading={isLoading}
                        error={fetchError?.error}
                        emptyMessage={fetchError ? fetchError.error : undefined}
                        currentSearch={currentSearch}
                      />
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
