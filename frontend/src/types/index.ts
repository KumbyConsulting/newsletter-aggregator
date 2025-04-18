/**
 * Represents the structure of a single article's metadata from the API.
 */
export interface ArticleMetadata {
  id: string;
  title: string;
  link: string;
  description: string; // Cleaned HTML or text
  pub_date: string; // Consider parsing this to Date object on frontend
  source: string;
  topic: string;
  summary?: string; // Optional AI-generated summary
  image_url?: string; // Optional image URL
  has_full_content: boolean;
  reading_time: number; // Estimated reading time in minutes
  relevance_score?: number; // Optional score for search results
  is_recent: boolean;
}

/**
 * Represents the structure of an article object as returned in lists.
 */
export interface Article {
  id: string;
  metadata: ArticleMetadata;
  // Potentially other top-level fields if the API structure changes
}

/**
 * Represents the response structure for the GET /api/articles endpoint.
 */
export interface ArticlesApiResponse {
  articles: Article[];
  total: number;
  page: number;
  total_pages: number;
  query_time?: number; // Time taken for the query in ms
}

/**
 * Represents the structure for a topic and its statistics.
 */
export interface TopicStat {
  topic: string;
  count: number;
  percentage: number;
  // Potentially add trend, recent_count, growth_rate if backend provides them
}

/**
 * Represents the response structure for the GET /api/topics endpoint.
 */
export interface TopicsApiResponse {
  topics: TopicStat[];
}

/**
 * Represents the structure of a single source returned by the RAG API.
 */
export interface RAGSource {
  title: string;
  source: string;
  date: string;
  link: string;
}

/**
 * Represents the structure of the non-streaming RAG API response.
 */
export interface RAGResponse {
  query?: string; // Echoed back query
  response: string; // The AI-generated text response
  sources: RAGSource[];
  timestamp: string; // Consider parsing to Date
  confidence: number;
  status?: string; // e.g., "success"
}

/**
 * Represents the structure of the streaming RAG API message (Server-Sent Event data).
 */
export interface RAGStreamEvent {
  chunk?: string; // Text chunk for intermediate messages
  done: boolean;
  sources?: RAGSource[]; // Sent with the final 'done: true' message
  full_response?: string; // Optional full response text with final message
  error?: string; // Sent if an error occurs
}

/**
 * Represents the status of the background article update process.
 */
export interface UpdateStatus {
  in_progress: boolean;
  last_update: number | null; // Timestamp (consider converting to Date)
  status: string; // e.g., "idle", "running", "completed", "failed"
  progress: number; // 0-100
  message: string;
  error: string | null;
  sources_processed: number;
  total_sources: number;
  articles_found: number;
  estimated_completion_time: number | null; // Timestamp for estimated completion
  can_be_cancelled: boolean; // Whether the update can be cancelled
}

/**
 * Represents the structure for search suggestions.
 */
export interface SearchSuggestionsResponse {
  suggestions: string[];
}

/**
 * Represents the structure for a single summary generation response.
 */
export interface SummarizeResponse {
  summary: string;
}

/**
 * Basic structure for API errors.
 */
export interface ApiErrorResponse {
  error: string;
  status?: string;
  status_code?: number;
  payload?: Record<string, any>;
  retry_after?: number; // For 429 errors
} 