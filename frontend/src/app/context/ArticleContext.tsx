'use client';

import { createContext, useContext, useReducer, ReactNode, useCallback } from 'react';
import { Article } from '@/types';

// Define state type
interface ArticleState {
  articles: Article[];
  totalArticles: number;
  currentPage: number;
  perPage: number;
  totalPages: number;
  isLoading: boolean;
  error: string | null;
  selectedTopic: string;
  searchQuery: string;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
}

// Initial state
const initialState: ArticleState = {
  articles: [],
  totalArticles: 0,
  currentPage: 1,
  perPage: 9,
  totalPages: 0,
  isLoading: false,
  error: null,
  selectedTopic: 'All',
  searchQuery: '',
  sortBy: 'pub_date',
  sortOrder: 'desc',
};

// Action types
type ArticleAction =
  | { type: 'FETCH_ARTICLES_START' }
  | { type: 'FETCH_ARTICLES_SUCCESS'; payload: { articles: Article[]; total: number; totalPages: number } }
  | { type: 'FETCH_ARTICLES_ERROR'; payload: string }
  | { type: 'SET_PAGE'; payload: number }
  | { type: 'SET_PER_PAGE'; payload: number }
  | { type: 'SET_TOPIC'; payload: string }
  | { type: 'SET_SEARCH_QUERY'; payload: string }
  | { type: 'SET_SORT'; payload: { sortBy: string; sortOrder: 'asc' | 'desc' } }
  | { type: 'RESET_FILTERS' };

// Reducer function
function articleReducer(state: ArticleState, action: ArticleAction): ArticleState {
  switch (action.type) {
    case 'FETCH_ARTICLES_START':
      return {
        ...state,
        isLoading: true,
        error: null,
      };
    case 'FETCH_ARTICLES_SUCCESS':
      return {
        ...state,
        articles: action.payload.articles,
        totalArticles: action.payload.total,
        totalPages: action.payload.totalPages,
        isLoading: false,
      };
    case 'FETCH_ARTICLES_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false,
      };
    case 'SET_PAGE':
      return {
        ...state,
        currentPage: action.payload,
      };
    case 'SET_PER_PAGE':
      return {
        ...state,
        perPage: action.payload,
        currentPage: 1, // Reset to first page when changing items per page
      };
    case 'SET_TOPIC':
      return {
        ...state,
        selectedTopic: action.payload,
        currentPage: 1, // Reset to first page when changing topic
      };
    case 'SET_SEARCH_QUERY':
      return {
        ...state,
        searchQuery: action.payload,
        currentPage: 1, // Reset to first page when searching
      };
    case 'SET_SORT':
      return {
        ...state,
        sortBy: action.payload.sortBy,
        sortOrder: action.payload.sortOrder,
        currentPage: 1, // Reset to first page when changing sort
      };
    case 'RESET_FILTERS':
      return {
        ...state,
        selectedTopic: 'All',
        searchQuery: '',
        sortBy: 'pub_date',
        sortOrder: 'desc',
        currentPage: 1,
      };
    default:
      return state;
  }
}

// Create context
interface ArticleContextType extends ArticleState {
  fetchArticles: () => Promise<void>;
  setPage: (page: number) => void;
  setPerPage: (perPage: number) => void;
  setTopic: (topic: string) => void;
  setSearchQuery: (query: string) => void;
  setSort: (sortBy: string, sortOrder: 'asc' | 'desc') => void;
  resetFilters: () => void;
}

const ArticleContext = createContext<ArticleContextType | undefined>(undefined);

// Provider component
interface ArticleProviderProps {
  children: ReactNode;
}

export function ArticleProvider({ children }: ArticleProviderProps) {
  const [state, dispatch] = useReducer(articleReducer, initialState);

  // Fetch articles from API
  const fetchArticles = useCallback(async () => {
    dispatch({ type: 'FETCH_ARTICLES_START' });
    
    try {
      // Dynamic import to avoid SSR issues
      const { getArticles } = await import('../services/api');
      
      const result = await getArticles(
        state.currentPage,
        state.perPage,
        state.searchQuery,
        state.selectedTopic,
        state.sortBy,
        state.sortOrder
      );
      
      dispatch({
        type: 'FETCH_ARTICLES_SUCCESS',
        payload: {
          articles: result.articles,
          total: result.total,
          totalPages: result.total_pages,
        },
      });
    } catch (error) {
      dispatch({
        type: 'FETCH_ARTICLES_ERROR',
        payload: error instanceof Error ? error.message : 'Failed to fetch articles',
      });
    }
  }, [
    state.currentPage,
    state.perPage,
    state.searchQuery,
    state.selectedTopic,
    state.sortBy,
    state.sortOrder,
  ]);

  // Action creators
  const setPage = useCallback((page: number) => {
    dispatch({ type: 'SET_PAGE', payload: page });
  }, []);

  const setPerPage = useCallback((perPage: number) => {
    dispatch({ type: 'SET_PER_PAGE', payload: perPage });
  }, []);

  const setTopic = useCallback((topic: string) => {
    dispatch({ type: 'SET_TOPIC', payload: topic });
  }, []);

  const setSearchQuery = useCallback((query: string) => {
    dispatch({ type: 'SET_SEARCH_QUERY', payload: query });
  }, []);

  const setSort = useCallback((sortBy: string, sortOrder: 'asc' | 'desc') => {
    dispatch({ type: 'SET_SORT', payload: { sortBy, sortOrder } });
  }, []);

  const resetFilters = useCallback(() => {
    dispatch({ type: 'RESET_FILTERS' });
  }, []);

  // Provide the context value
  const value = {
    ...state,
    fetchArticles,
    setPage,
    setPerPage,
    setTopic,
    setSearchQuery,
    setSort,
    resetFilters,
  };

  return <ArticleContext.Provider value={value}>{children}</ArticleContext.Provider>;
}

// Custom hook to use the context
export function useArticles() {
  const context = useContext(ArticleContext);
  
  if (context === undefined) {
    throw new Error('useArticles must be used within an ArticleProvider');
  }
  
  return context;
} 