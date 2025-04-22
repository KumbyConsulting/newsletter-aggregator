/**
 * Enhanced API Service with better caching
 * 
 * Improved caching and throttling to reduce unnecessary API calls
 */

// Configure API base URL to use API Gateway
const API_GATEWAY_URL = process.env.NEXT_PUBLIC_API_GATEWAY_URL || 'https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev';
const API_BASE_URL = `${API_GATEWAY_URL}/api`; // Append /api to match OpenAPI spec paths
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';

// Add throttling and caching utilities
import debounce from 'lodash/debounce';
import throttle from 'lodash/throttle';

import {
  Article as CoreArticle,
  ArticleMetadata,
  TopicStat as TopicStatType,
  UpdateStatus as UpdateStatusType,
  RAGResponse as RAGResponseType,
  ApiErrorResponse,
  RAGStreamEvent,
  ArticlesApiResponse,
  TopicsApiResponse
} from '@/types';

// Topic Statistics
export interface TopicStats {
  topic: string;
  count: number;
  percentage: number;
  trend?: 'up' | 'down' | 'stable';
  growth_rate?: number;
  recent_count?: number;
}

// API Types
export type AnalysisType = 
  | 'regulatory'
  | 'safety'
  | 'clinical'
  | 'pipeline'
  | 'market'
  | 'competitive'
  | 'manufacturing'
  | 'digital';

// Local Article Interface for API calls
export interface Article {
  id: string;
  title: string;
  description: string;
  link: string;
  pub_date: string;
  source: string;
  topic: string;
  summary?: string;
  image_url?: string;
  reading_time?: number;
  has_full_content?: boolean;
  date?: string;
}

// Mapped Article Type for compatibility with CoreArticle
export interface MappedArticle {
  id: string;
  metadata: ArticleMetadata;
}

// Validation Result Interface
export interface ValidationResult {
  results: Record<string, boolean>;
  count: {
    total: number;
    valid: number;
    invalid: number;
  }
}

// Update Status Interface
export interface UpdateStatus extends UpdateStatusType {}

// Scraping Metrics Interface
export interface ScrapingMetrics {
  duration_seconds: number;
  total_feeds: number;
  successful_feeds: number;
  failed_feeds: number;
  total_articles: number;
  matched_articles: number;
  summary_success_rate: string;
  cache_hit_rate: string;
  rate_limits: number;
}

// RAG Query Interface
export interface RAGQuery {
  query: string;
  time_aware?: boolean;
  analysis_type?: AnalysisType;
}

// RAG Response Interface
export interface RAGResponse extends RAGResponseType {}

// RAG History Item Interface
export interface RAGHistoryItem {
  id: string;
  query: string;
  response: string;
  timestamp: string;
  analysis_type?: AnalysisType;
  confidence?: number;
  sources?: Article[];
}

// Add new SavedAnalysis interface below the existing interfaces
export interface SavedAnalysis {
  id: string;
  query: string;
  response: string;
  timestamp: string;
  analysis_type: AnalysisType;
  confidence: number;
  sources: Article[];
}

// Enhanced cache implementation with TTL
interface CacheItem<T> {
  data: T;
  timestamp: number;
  expiry: number;
}

class APICache {
  private cache: Map<string, CacheItem<any>> = new Map();
  private pendingRequests: Map<string, Promise<any>> = new Map();
  
  constructor(private defaultTTL: number = 60000) {} // Default TTL is 1 minute

  // Get data from cache
  get<T>(key: string): T | null {
    const item = this.cache.get(key);
    
    if (!item) return null;
    
    // Check if item has expired
    if (Date.now() > item.expiry) {
      this.cache.delete(key);
      return null;
    }
    
    return item.data as T;
  }
  
  // Store data in cache
  set<T>(key: string, data: T, ttl: number = this.defaultTTL): void {
    const timestamp = Date.now();
    this.cache.set(key, {
      data,
      timestamp,
      expiry: timestamp + ttl
    });
  }
  
  // Check if a request is pending
  isPending(key: string): boolean {
    return this.pendingRequests.has(key);
  }
  
  // Get a pending request
  getPending<T>(key: string): Promise<T> | null {
    return (this.pendingRequests.get(key) as Promise<T>) || null;
  }
  
  // Set a pending request
  setPending<T>(key: string, promise: Promise<T>): void {
    this.pendingRequests.set(key, promise);
    
    // Clean up once resolved
    promise.finally(() => {
      this.pendingRequests.delete(key);
    });
  }
  
  // Clear the cache
  clear(): void {
    this.cache.clear();
  }
  
  // Clear a specific key
  clearKey(key: string): void {
    this.cache.delete(key);
  }
}

// Initialize the API cache
const apiCache = new APICache();

// Add retry utility
const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const retryWithBackoff = async <T>(
  fn: () => Promise<T>,
  retries = 3,
  initialDelay = 1000,
  maxDelay = 10000,
  factor = 2
): Promise<T> => {
  let attempt = 0;
  let delay = initialDelay;

  while (attempt < retries) {
    try {
      return await fn();
    } catch (error) {
      attempt++;
      if (attempt === retries) {
        throw error;
      }

      // Only retry on timeout or network errors
      if (error instanceof Error && 
          !((error as any).isTimeout || 
            error.message.includes('Failed to fetch') ||
            error.message.includes('NetworkError'))) {
        throw error;
      }

      console.warn(`Attempt ${attempt} failed, retrying in ${delay}ms...`);
      await wait(delay);
      delay = Math.min(delay * factor, maxDelay);
    }
  }

  throw new Error('Max retries reached');
};

/**
 * Make a request to the API with the configured base URL
 * Enhanced with caching and better error handling
 */
async function apiRequest<T>(
  endpoint: string, 
  options: RequestInit = {}, 
  cacheKey: string = '', 
  cacheTTL: number = 60000,
  timeoutMs: number = 30000,
  retryConfig = {
    retries: 3,
    initialDelay: 1000,
    maxDelay: 10000,
    factor: 2
  }
): Promise<T> {
  const url = new URL(`${API_BASE_URL}${endpoint}`);
  
  // Set up headers according to OpenAPI spec
  const headers = {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY,
    'Accept': 'application/json',
    'Origin': window.location.origin,
    ...options.headers,
  };

  const actualTimeout = endpoint === '/update/status' ? 10000 : timeoutMs;
  
  const requestPromise = retryWithBackoff(
    async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(new Error('TimeoutError: signal timed out')), actualTimeout);
      
      try {
        const response = await fetch(url.toString(), {
          ...options,
          headers,
          mode: 'cors',
          credentials: 'omit',
          signal: controller.signal,
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          let errorData: ApiErrorResponse;
          
          try {
            errorData = await response.json();
          } catch (e) {
            errorData = {
              error: `Server error (${response.status})`,
              status_code: response.status
            };
          }
          
          // Enhanced error handling for CORS and auth issues
          if (response.status === 403) {
            errorData.error = 'API key invalid or missing';
          } else if (response.status === 0 || !response.status) {
            errorData.error = 'Network error - CORS or connectivity issue';
            console.error('CORS Error Details:', {
              origin: window.location.origin,
              target: url.toString(),
              status: response.status,
              statusText: response.statusText
            });
          }
          
          const error = new Error(errorData.error || `API request failed with status ${response.status}`);
          (error as any).statusCode = response.status;
          (error as any).apiError = errorData;
          
          if (errorData.error) {
            (error as any).error = errorData.error;
          }
          if (errorData.status_code) {
            (error as any).status_code = errorData.status_code;
          }
          
          throw error;
        }
        
        if (response.status === 204) {
          return {} as T;
        }
        
        let data: T;
        try {
          data = await response.json() as T;
        } catch (e) {
          console.warn('Invalid JSON response from API:', e);
          data = {} as T;
        }
        
        apiCache.set<T>(cacheKey, data, cacheTTL);
        
        return data;
      } finally {
        clearTimeout(timeoutId);
      }
    },
    retryConfig.retries,
    retryConfig.initialDelay,
    retryConfig.maxDelay,
    retryConfig.factor
  );
  
  apiCache.setPending<T>(cacheKey, requestPromise);
  
  return requestPromise;
}

// Article cache TTLs (milliseconds)
const CACHE_TTLS = {
  ARTICLES: 60000,      // 1 minute
  TOPICS: 300000,       // 5 minutes
  TOPICS_STATS: 300000, // 5 minutes
  SEARCH: 30000,        // 30 seconds
  SIMILAR: 300000,      // 5 minutes
};

/**
 * Get topic statistics with enhanced caching
 */
export async function getTopicStats(): Promise<{ topics: TopicStats[] }> {
  return apiRequest<{ topics: TopicStats[] }>(
    '/topics/stats',
    {},
    'topic_stats',
    CACHE_TTLS.TOPICS_STATS
  );
}

/**
 * Get update status
 */
export async function getUpdateStatus(): Promise<UpdateStatus> {
  try {
    return await apiRequest<UpdateStatus>(
      '/update/status',
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      },
      'update_status',
      5000, // Cache for 5 seconds
      10000, // Timeout after 10 seconds
      {
        retries: 2,
        initialDelay: 1000,
        maxDelay: 3000,
        factor: 1.5
      }
    );
  } catch (error) {
    console.warn('Server unavailable during status check, using fallback status');
    // Return a fallback status object
    return {
      in_progress: false,
      status: 'idle',
      progress: 0,
      message: 'Status check failed, please try again',
      sources_processed: 0,
      total_sources: 0,
      articles_found: 0,
      last_update: null,
      error: error instanceof Error ? error.message : 'Unknown error',
      estimated_completion_time: null,
      can_be_cancelled: false
    };
  }
}

/**
 * Get scraping metrics
 * Returns metrics about the last scraping operation
 */
export async function getScrapingMetrics(): Promise<ScrapingMetrics> {
  // Try to get this information from the update status endpoint
  const status = await getUpdateStatus();
  
  // Extract metrics from the status if available
  return {
    duration_seconds: 0, // Placeholder
    total_feeds: status.total_sources || 0,
    successful_feeds: status.sources_processed || 0,
    failed_feeds: status.total_sources ? status.total_sources - status.sources_processed : 0,
    total_articles: status.articles_found || 0,
    matched_articles: status.articles_found || 0, // Assuming all found articles are matched
    summary_success_rate: "0%", // Placeholder
    cache_hit_rate: "0%", // Placeholder
    rate_limits: 0 // Placeholder
  };
}

/**
 * Start update process
 */
export async function startUpdate(): Promise<{ success: boolean; message: string; status: UpdateStatus }> {
  return apiRequest('/update/start', {
    method: 'POST',
  });
}

/**
 * Poll for updates
 * Continuously poll the update status API until a condition is met
 * Uses exponential backoff to reduce server load
 * @param initialIntervalMs - Initial poll interval in milliseconds
 * @param maxIntervalMs - Maximum poll interval in milliseconds
 * @param maxAttempts - Maximum number of attempts before giving up
 * @param stopCondition - Function that determines when to stop polling
 */
export async function pollUpdateStatus(
  initialIntervalMs: number = 2000,
  maxIntervalMs: number = 15000,
  maxAttempts: number = 100,
  stopCondition: (status: UpdateStatus) => boolean = (status) => !status.in_progress
): Promise<UpdateStatus> {
  let attempts = 0;
  let currentInterval = initialIntervalMs;
  
  while (attempts < maxAttempts) {
    attempts++;
    
    try {
      const status = await getUpdateStatus();
      
      if (stopCondition(status)) {
        return status;
      }
      
      // Calculate next interval using exponential backoff with progress-based adjustment
      // As progress increases, polling frequency decreases
      if (status.progress) {
        // Adjust interval based on progress: higher progress = longer interval
        const progressFactor = Math.min(1, status.progress / 100);
        currentInterval = Math.min(
          maxIntervalMs,
          initialIntervalMs * (1 + progressFactor * 3)
        );
      } else {
        // Standard exponential backoff if no progress info
        currentInterval = Math.min(
          maxIntervalMs,
          currentInterval * 1.2
        );
      }
      
      // Wait for the calculated interval
      await new Promise(resolve => setTimeout(resolve, currentInterval));
    } catch (error) {
      console.error('Error polling update status:', error);
      // On error, wait a bit longer before retrying
      await new Promise(resolve => setTimeout(resolve, currentInterval * 2));
    }
  }
  
  throw new Error('Max polling attempts reached');
}

/**
 * Get articles with filtering and pagination
 * Enhanced with better caching using search parameters
 */
export async function getArticles(
  page: number = 1,
  perPage: number = 9,
  searchQuery: string = '',
  topic: string = 'All',
  sortBy: string = 'pub_date',
  sortOrder: string = 'desc',
  source: string = '',
  dateFrom: string = '',
  dateTo: string = '',
  readingTime: string = '',
  hasSummary: boolean = false,
  hasFullContent: boolean = false,
  searchType?: string,
  threshold?: number,
  fields?: string[]
): Promise<{
  articles: CoreArticle[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
  topics: string[];
}> {
  // Build query parameters
  const params = new URLSearchParams({
    limit: perPage.toString(),
    page: page.toString(),
    sort_by: sortBy,
    sort_order: sortOrder
  });
  
  if (searchQuery) {
    params.append('search', searchQuery);
    if (searchType && searchType !== 'auto') {
      params.append('searchType', searchType);
    }
    if (threshold !== undefined && threshold !== null) {
      params.append('threshold', threshold.toString());
    }
    if (fields && fields.length > 0) {
      params.append('fields', fields.join(','));
    }
  }
  
  if (topic !== 'All') {
    params.append('topic', topic);
  }
  
  if (source) {
    params.append('source', source);
  }
  
  if (dateFrom) {
    params.append('date_from', dateFrom);
  }
  
  if (dateTo) {
    params.append('date_to', dateTo);
  }
  
  if (readingTime) {
    params.append('reading_time', readingTime);
  }
  
  if (hasSummary) {
    params.append('has_summary', '1');
  }
  
  if (hasFullContent) {
    params.append('has_full_content', '1');
  }
  
  // Create a cache key based on the query parameters
  const cacheKey = `articles_${params.toString()}`;
  
  // Use a shorter cache TTL for search queries
  const cacheTTL = searchQuery ? CACHE_TTLS.SEARCH : CACHE_TTLS.ARTICLES;
  
  // Fetch articles from API with caching - use Next.js API route now
  const data = await apiRequest<{
    articles: CoreArticle[];
    total: number;
    topics: string[];
    error?: string;
  }>(
    `/articles?${params.toString()}`,
    {},
    cacheKey,
    cacheTTL
  );
  
  if (data.error) {
    throw new Error(data.error);
  }
  
  // Return the articles in their original format - they already have the structure
  // { id, metadata } that the components expect
  return {
    articles: data.articles,
    total: data.total || 0,
    page: page,
    per_page: perPage,
    total_pages: data.total ? Math.ceil(data.total / perPage) : 1,
    topics: data.topics || []
  };
}

/**
 * Get similar articles by ID
 */
export async function getSimilarArticles(articleId: string): Promise<CoreArticle[]> {
  const data = await apiRequest<{
    articles: CoreArticle[];
  }>(`/similar-articles/${articleId}`);
  
  // Return the articles directly without transformation
  return data.articles || [];
}

/**
 * Submit a RAG query and get a response
 */
export async function submitRagQuery(query: RAGQuery): Promise<RAGResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/rag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(query),
      // Add timeout to prevent hanging requests
      signal: AbortSignal.timeout(30000) // 30 second timeout
    });
    
    if (!response.ok) {
      let errorData = { error: 'Unknown error' };
      try {
        errorData = await response.json();
      } catch (e) {
        // If JSON parsing fails, create a more specific error based on status
        if (response.status === 429) {
          errorData = { error: 'Rate limit exceeded. Please try again later.' };
        } else if (response.status === 504) {
          errorData = { error: 'Request timed out. Please try a more specific query.' };
        } else if (response.status >= 500) {
          errorData = { error: 'Server error. The AI service might be temporarily unavailable.' };
        }
      }
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }
    
    return response.json();
  } catch (error) {
    // Handle network errors or other exceptions
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      throw new Error('Network error. Please check your connection and try again.');
    }
    throw error;
  }
}

/**
 * Stream a RAG query response
 */
export async function streamRagQuery(
  query: RAGQuery,
  onChunk: (chunk: string) => void
): Promise<void> {
  try {
    const url = new URL(`${API_BASE_URL}/rag/stream`);
    url.searchParams.append('key', API_KEY);

    const response = await fetch(url.toString(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(query),
      // Add timeout to prevent hanging requests
      signal: AbortSignal.timeout(60000) // 60 second timeout for streaming
    });
    
    if (!response.ok) {
      let errorData = { error: 'Failed to stream query' };
      try {
        errorData = await response.json();
      } catch (e) {
        // If JSON parsing fails, create a more specific error based on status
        if (response.status === 429) {
          errorData = { error: 'Rate limit exceeded. Please try again later.' };
        } else if (response.status === 504 || response.status === 408) {
          errorData = { error: 'Request timed out. Please try a more specific query.' };
        } else if (response.status >= 500) {
          errorData = { error: 'Server error. The AI service might be temporarily unavailable.' };
        }
      }
      throw new Error(errorData.error || `Stream error: ${response.status}`);
    }
    
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Stream not available');
    }
    
    const decoder = new TextDecoder();
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        onChunk(chunk);
      }
    } catch (readError) {
      console.error('Error reading from stream:', readError);
      throw new Error('Stream was interrupted. Please try again.');
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      throw new Error('Network error. Please check your connection and try again.');
    } else if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Request timed out. The server took too long to respond.');
    }
    throw error;
  }
}

/**
 * Get RAG query history
 */
export async function getRagHistory(): Promise<RAGHistoryItem[]> {
  const response = await fetch('/api/rag/history');
  
  if (!response.ok) {
    throw new Error('Failed to get history');
  }
  
  return response.json();
}

/**
 * Clear RAG query history
 */
export async function clearRagHistory(): Promise<void> {
  const response = await fetch('/api/rag/history', {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error('Failed to clear history');
  }
}

/**
 * Backup data
 */
export async function backupData(): Promise<{ success: boolean; backup_file: string }> {
  return apiRequest('/backup', {
    method: 'POST'
  });
}

/**
 * List backups
 */
export async function listBackups(): Promise<{ backups: string[] }> {
  return apiRequest('/backups');
}

/**
 * Restore backup
 */
export async function restoreBackup(backupFile: string): Promise<{ success: boolean; message: string }> {
  return apiRequest('/restore', {
    method: 'POST',
    body: JSON.stringify({ backup_file: backupFile })
  });
}

/**
 * Generate summaries for articles that don't have them
 */
export async function generateSummaries(): Promise<{ success: boolean; message: string; count: number }> {
  return apiRequest('/generate-summaries', {
    method: 'POST'
  });
}

/**
 * Get sources with counts
 */
export async function getSources(): Promise<{ sources: { name: string; url: string; count: number; logo_url?: string; description?: string; }[] }> {
  // Use the apiRequest helper with proper caching
  return apiRequest<{ sources: { name: string; url: string; count: number; logo_url?: string; description?: string; }[] }>(
    '/sources',
    {},
    'sources_list',
    CACHE_TTLS.TOPICS // Reuse the topics cache TTL (5 minutes)
  );
}

/**
 * Get search suggestions
 */
export async function getSearchSuggestions(query: string): Promise<{ suggestions: string[] }> {
  return apiRequest(`/search/suggestions?q=${encodeURIComponent(query)}`);
}

/**
 * Save analysis results
 */
export async function saveAnalysis(analysisData: {
  query: string;
  response: string;
  sources: Article[];
  confidence: number;
  analysis_type: AnalysisType;
  timestamp: string;
}): Promise<{ success: boolean; id: string }> {
  return apiRequest('/analysis/save', {
    method: 'POST',
    body: JSON.stringify(analysisData)
  });
}

/**
 * Get saved analyses
 */
export async function getSavedAnalyses(
  limit: number = 10,
  offset: number = 0,
  sortBy: string = 'timestamp',
  sortOrder: string = 'desc'
): Promise<{
  analyses: SavedAnalysis[];
  total: number;
  limit: number;
  offset: number;
}> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
    sort_by: sortBy,
    sort_order: sortOrder
  });
  
  return apiRequest(`/analysis/saved?${params.toString()}`);
}

/**
 * Get a specific saved analysis by ID
 */
export async function getSavedAnalysis(analysisId: string): Promise<SavedAnalysis> {
  return apiRequest(`/analysis/saved/${analysisId}`);
}

/**
 * Delete a saved analysis by ID
 */
export async function deleteSavedAnalysis(analysisId: string): Promise<{ success: boolean; message: string }> {
  return apiRequest(`/analysis/saved/${analysisId}`, {
    method: 'DELETE'
  });
}

/**
 * Validate multiple article IDs to check if they exist in the database
 * Helps ensure frontend and backend are in sync
 * Enhanced with caching, throttling and duplicate request prevention
 */
export const validateArticles = throttle(async function(articleIds: string[]): Promise<ValidationResult> {
  // Sort IDs to ensure consistent cache keys regardless of array order
  const sortedIds = [...articleIds].sort();
  const cacheKey = sortedIds.join(',');
  
  // Check if we already have a pending request for these IDs
  if (apiCache.isPending(cacheKey)) {
    return apiCache.getPending<ValidationResult>(cacheKey) as Promise<ValidationResult>;
  }
  
  // Check if we have cached results that are still valid
  const now = Date.now();
  const cachedResults: Record<string, boolean> = {};
  let allCached = sortedIds.length > 0;
  
  for (const id of sortedIds) {
    const cachedValue = apiCache.get<boolean>(`validated_${id}`);
    if (cachedValue !== null) {
      cachedResults[id] = cachedValue;
    } else {
      allCached = false;
      break;
    }
  }
  
  // If all IDs have valid cached results, return them
  if (allCached) {
    const validCount = Object.values(cachedResults).filter(Boolean).length;
    return {
      results: cachedResults,
      count: {
        total: sortedIds.length,
        valid: validCount,
        invalid: sortedIds.length - validCount
      }
    };
  }
  
  // Otherwise, make the API call
  const validationPromise = new Promise<ValidationResult>(async (resolve) => {
    try {
      // Don't make API calls for empty arrays
      if (sortedIds.length === 0) {
        resolve({
          results: {},
          count: { total: 0, valid: 0, invalid: 0 }
        });
        return;
      }
      
      // Use the Next.js API route instead of directly calling the backend
      const response = await fetch(`/api/articles/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ article_ids: sortedIds }),
        // Add timeout to prevent hanging requests
        signal: AbortSignal.timeout(10000) // 10 second timeout
      });

      if (!response.ok) {
        throw new Error(`Validation failed: ${response.status}`);
      }

      const data = await response.json();
      
      // Cache individual results
      for (const [id, isValid] of Object.entries(data.results)) {
        apiCache.set<boolean>(`validated_${id}`, isValid as boolean);
      }
      
      resolve(data);
    } catch (error) {
      console.error('Error validating articles:', error);
      
      // Create fallback response - assume all articles are valid to prevent disappearing content
      const fallbackResults: Record<string, boolean> = {};
      sortedIds.forEach(id => {
        // If we have a cached value, use it even if expired
        const cachedValue = apiCache.get<boolean>(`validated_${id}`);
        if (cachedValue !== null) {
          fallbackResults[id] = cachedValue;
        } else {
          // Assume valid in case of error - better to show potentially stale content than nothing
          fallbackResults[id] = true;
          // Cache the fallback value with a shorter expiry
          apiCache.set<boolean>(`validated_${id}`, true, 30000); // Expire in 30 seconds
        }
      });
      
      resolve({
        results: fallbackResults,
        count: {
          total: sortedIds.length,
          valid: Object.values(fallbackResults).filter(Boolean).length,
          invalid: sortedIds.length - Object.values(fallbackResults).filter(Boolean).length
        }
      });
    } finally {
      // Remove from pending validations
      setTimeout(() => {
        apiCache.clearKey(cacheKey);
      }, 0);
    }
  });
  
  // Store the pending promise
  apiCache.setPending<ValidationResult>(cacheKey, validationPromise);
  return validationPromise;
}, 2000); // Throttle to at most one call every 2 seconds

/**
 * Get available topics with caching
 */
export async function getTopics(): Promise<string[]> {
  try {
    const data = await apiRequest<{ topics: string[] }>(
      '/topics',
      {},
      'topics_list',
      CACHE_TTLS.TOPICS
    );
    return data.topics || [];
  } catch (error) {
    console.error('Error fetching topics:', error);
    return [];
  }
}

/**
 * Clear API cache for specific endpoints or all
 */
export function clearCache(endpoint?: string): void {
  if (endpoint) {
    apiCache.clearKey(endpoint);
  } else {
    apiCache.clear();
  }
}

/**
 * Convert between Article formats
 * Helps with compatibility between API service and component expectations
 * 
 * Note: This function is kept for backward compatibility but is now less frequently used
 * as we're directly using the CoreArticle format from the API
 */
export function mapArticleToCore(article: Article): MappedArticle {
  return {
    id: article.id,
    metadata: {
      id: article.id,
      title: article.title,
      description: article.description,
      link: article.link,
      pub_date: article.pub_date,
      source: article.source,
      topic: article.topic,
      summary: article.summary,
      image_url: article.image_url,
      reading_time: article.reading_time || 0,
      has_full_content: article.has_full_content || false,
      is_recent: new Date(article.pub_date) > new Date(Date.now() - 24 * 60 * 60 * 1000)
    }
  };
}

/**
 * Convert from CoreArticle to local Article format
 * 
 * Note: This function is kept for backward compatibility but is now less frequently used
 * as we're directly using the CoreArticle format from the API
 */
export function mapCoreToArticle(coreArticle: CoreArticle): Article {
  const { id, metadata } = coreArticle;
  return {
    id,
    title: metadata.title,
    description: metadata.description,
    link: metadata.link,
    pub_date: metadata.pub_date,
    source: metadata.source,
    topic: metadata.topic,
    summary: metadata.summary,
    image_url: metadata.image_url,
    reading_time: metadata.reading_time,
    has_full_content: metadata.has_full_content
  };
}

// Fix boolean | null type issues by adding null checks where needed
export async function getArticle(id: string): Promise<Article> {
  const cachedArticle = apiCache.get<Article>(`article_${id}`);
  if (cachedArticle) return cachedArticle;
  
  return apiRequest<Article>(`/articles/${id}`);
}

/**
 * Adapter function to make the getArticles function match the interface expected by page.tsx
 * This bridges the gap between the two API implementations
 */
export async function fetchArticles(params: {
  page?: number;
  limit?: number;
  topic?: string;
  search?: string;
  sort_by?: string;
  sort_order?: string;
  searchType?: string;
  threshold?: number;
  fields?: string[];
}): Promise<ArticlesApiResponse> {
  try {
    const results = await getArticles(
      params.page || 1,             // page
      params.limit || 10,           // perPage
      params.search || '',          // searchQuery
      params.topic !== 'All' ? params.topic || '' : '', // topic (adjusted)
      params.sort_by || 'pub_date', // sortBy
      params.sort_order || 'desc',  // sortOrder
      '',                            // source (default)
      '',                            // dateFrom (default)
      '',                            // dateTo (default)
      '',                            // readingTime (default)
      false,                         // hasSummary (default)
      false,                         // hasFullContent (default)
      params.searchType,             // searchType (optional)
      params.threshold,              // threshold (optional)
      params.fields                  // fields (optional)
    );

    return {
      articles: results.articles,
      page: results.page,
      total: results.total,
      total_pages: results.total_pages
    };
  } catch (error) {
    console.error('Error fetching articles via adapter:', error); // Added adapter specific logging
    // Rethrow the error to be handled by the caller
    if (error instanceof Error) {
      throw new Error(`Failed to fetch articles: ${error.message}`);
    } else {
      throw new Error('An unknown error occurred while fetching articles.');
    }
  }
}

/**
 * Adapter function to make the getTopicStats function match the interface expected by page.tsx
 */
export async function fetchTopics(): Promise<TopicsApiResponse> {
  try {
    const stats = await getTopicStats();
    return {
      topics: stats.topics
    };
  } catch (error) {
    console.error('Error fetching topics:', error);
    throw error;
  }
} 