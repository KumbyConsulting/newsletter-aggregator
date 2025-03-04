/**
 * API Service
 * 
 * Centralized API service for making calls to the backend API
 */

// Topic Statistics
export interface TopicStats {
  topic: string;
  count: number;
  percentage: number;
}

// Article Interface
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
}

// Update Status Interface
export interface UpdateStatus {
  in_progress: boolean;
  last_update: number | null;
  status: string;
  progress: number;
  message: string;
  error: string | null;
  sources_processed: number;
  total_sources: number;
  articles_found: number;
}

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

/**
 * Get topic statistics
 */
export async function getTopicStats(): Promise<{ topics: TopicStats[] }> {
  const response = await fetch('/api/topics');
  
  if (!response.ok) {
    throw new Error('Failed to fetch topic statistics');
  }
  
  return await response.json();
}

/**
 * Get update status
 */
export async function getUpdateStatus(): Promise<UpdateStatus> {
  const response = await fetch('/api/update/status');
  
  if (!response.ok) {
    throw new Error('Failed to fetch update status');
  }
  
  return await response.json();
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
  const response = await fetch('/api/update/start', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  });
  
  if (!response.ok) {
    throw new Error('Failed to start update');
  }
  
  return await response.json();
}

/**
 * Poll for updates
 * Continuously poll the update status API until a condition is met
 * @param intervalMs - Poll interval in milliseconds
 * @param maxAttempts - Maximum number of attempts before giving up
 * @param stopCondition - Function that determines when to stop polling
 */
export async function pollUpdateStatus(
  intervalMs: number = 5000,
  maxAttempts: number = 60,
  stopCondition: (status: UpdateStatus) => boolean = (status) => !status.in_progress
): Promise<UpdateStatus> {
  let attempts = 0;
  
  while (attempts < maxAttempts) {
    attempts++;
    
    try {
      const status = await getUpdateStatus();
      
      if (stopCondition(status)) {
        return status;
      }
      
      // Wait for the specified interval
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    } catch (error) {
      console.error('Error polling update status:', error);
      throw error;
    }
  }
  
  throw new Error('Max polling attempts reached');
}

/**
 * Get articles with filtering and pagination
 */
export async function getArticles(
  page: number = 1,
  perPage: number = 9,
  searchQuery: string = '',
  topic: string = 'All',
  sortBy: string = 'pub_date',
  sortOrder: string = 'desc'
): Promise<{
  articles: Article[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
  topics: string[];
}> {
  // Build query parameters
  const params = new URLSearchParams({
    limit: perPage.toString(),
    page: page.toString()
  });
  
  if (searchQuery) {
    params.append('search', searchQuery);
  }
  
  if (topic !== 'All') {
    params.append('topic', topic);
  }
  
  // Fetch articles from API
  const response = await fetch(`/api/articles?${params.toString()}`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch articles');
  }
  
  const data = await response.json();
  
  if (data.error) {
    throw new Error(data.error);
  }
  
  // Transform the nested article structure into the expected format
  const transformedArticles: Article[] = (data.articles || []).map((article: any) => {
    const metadata = article.metadata || {};
    return {
      id: article.id || metadata.id || '',
      title: metadata.title || 'Untitled',
      description: metadata.description || '',
      link: metadata.link || '#',
      pub_date: metadata.pub_date || 'Unknown date',
      source: metadata.source || 'Unknown source',
      topic: metadata.topic || 'Uncategorized',
      summary: metadata.summary || undefined,
      image_url: metadata.image_url || undefined
    };
  });
  
  return {
    articles: transformedArticles,
    total: data.count || transformedArticles.length,
    page: page,
    per_page: perPage,
    total_pages: Math.ceil((data.count || transformedArticles.length) / perPage),
    topics: data.topics || []
  };
}

/**
 * Get similar articles by ID
 */
export async function getSimilarArticles(articleId: string): Promise<Article[]> {
  const response = await fetch(`/api/similar-articles/${articleId}`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch similar articles');
  }
  
  const data = await response.json();
  
  // Transform the nested article structure into the expected format
  return (data.articles || []).map((article: any) => {
    const metadata = article.metadata || {};
    return {
      id: article.id || metadata.id || '',
      title: metadata.title || 'Untitled',
      description: metadata.description || '',
      link: metadata.link || '#',
      pub_date: metadata.pub_date || 'Unknown date',
      source: metadata.source || 'Unknown source',
      topic: metadata.topic || 'Uncategorized',
      summary: metadata.summary || undefined,
      image_url: metadata.image_url || undefined
    };
  });
} 