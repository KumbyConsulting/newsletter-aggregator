'use client';

import { Article as CoreArticle } from '@/types';
import { validateArticles, ValidationResult } from './api';
import debounce from 'lodash/debounce';

const LOCAL_STORAGE_KEY = 'article_validity_cache';
const VALIDITY_CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours
const BATCH_SIZE = 50; // Number of articles to validate in a single batch

interface ValidityCache {
  timestamp: number;
  results: Record<string, boolean>;
}

/**
 * Database Synchronization Service
 * 
 * Ensures that article data is consistent between frontend and backend
 * by validating article IDs and maintaining a local cache of valid articles.
 */
export class SyncService {
  private static instance: SyncService;
  private validityCache: ValidityCache = { timestamp: 0, results: {} };
  private syncInProgress: boolean = false;
  private pendingSyncRequests: Set<string> = new Set();
  
  private constructor() {
    this.loadValidityCache();
  }
  
  /**
   * Get singleton instance of SyncService
   */
  public static getInstance(): SyncService {
    if (!SyncService.instance) {
      SyncService.instance = new SyncService();
    }
    return SyncService.instance;
  }
  
  /**
   * Load article validity cache from localStorage
   */
  private loadValidityCache(): void {
    try {
      if (typeof window === 'undefined') return;
      
      const cachedData = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (cachedData) {
        const parsed = JSON.parse(cachedData) as ValidityCache;
        
        // Check if cache is still valid (not expired)
        const now = Date.now();
        if (now - parsed.timestamp < VALIDITY_CACHE_TTL) {
          this.validityCache = parsed;
          console.log(`Loaded validity cache with ${Object.keys(parsed.results).length} entries`);
        } else {
          console.log('Validity cache expired, will be refreshed');
          // Clear expired cache
          localStorage.removeItem(LOCAL_STORAGE_KEY);
          this.validityCache = { timestamp: 0, results: {} };
        }
      }
    } catch (error) {
      console.error('Error loading article validity cache:', error);
      this.validityCache = { timestamp: 0, results: {} };
    }
  }
  
  /**
   * Save article validity cache to localStorage
   */
  private saveValidityCache(): void {
    try {
      if (typeof window === 'undefined') return;
      
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(this.validityCache));
    } catch (error) {
      console.error('Error saving article validity cache:', error);
    }
  }
  
  /**
   * Check if an article ID is valid (exists in backend)
   * Uses local cache if available, otherwise falls back to API call
   */
  public async isArticleValid(articleId: string): Promise<boolean> {
    // Check cache first
    if (this.validityCache.results[articleId] !== undefined) {
      return this.validityCache.results[articleId];
    }
    
    // Not in cache, validate with API
    try {
      const validation = await validateArticles([articleId]);
      
      // Update cache with result
      this.validityCache.results[articleId] = validation.results[articleId] || false;
      this.validityCache.timestamp = Date.now();
      this.saveValidityCache();
      
      return validation.results[articleId] || false;
    } catch (error) {
      console.error('Error validating article:', error);
      return false;
    }
  }
  
  /**
   * Debounced method to save cache
   * Prevents multiple rapid writes to localStorage
   */
  private debouncedSaveCache = debounce(() => {
    this.saveValidityCache();
  }, 1000, { maxWait: 5000 });
  
  /**
   * Filter a list of articles to only include valid ones (exist in backend)
   * Efficiently validates in batches to reduce API calls
   * Optimized with caching and debouncing
   */
  public async filterValidArticles<T extends { id: string }>(articles: T[]): Promise<T[]> {
    if (!articles.length) return [];
    
    // Extract all article IDs and check which ones are in cache
    const allIds = articles.map(article => article.id);
    const cachedIds = new Set(Object.keys(this.validityCache.results));
    
    // Check which IDs are not in cache
    const idsToValidate = allIds.filter(id => !cachedIds.has(id));
    
    // If all IDs are in cache, we can return filtered results immediately
    if (idsToValidate.length === 0) {
      return articles.filter(article => this.validityCache.results[article.id] === true);
    }
    
    // Generate a unique request ID for this validation batch
    const requestId = Date.now().toString();
    this.pendingSyncRequests.add(requestId);
    
    // CRITICAL FIX: If another validation is in progress or if we encounter issues,
    // assume articles are valid to prevent disappearing content
    if (this.syncInProgress) {
      console.log('Another validation in progress, assuming all articles valid for now');
      // Return all articles as-is instead of filtering
      return [...articles];
    }
    
    this.syncInProgress = true;
    console.log(`Validating ${idsToValidate.length} uncached article IDs`);
    
    try {
      // Validate in batches to avoid overloading API
      for (let i = 0; i < idsToValidate.length; i += BATCH_SIZE) {
        // Check if this request was superseded by a newer one
        if (!this.pendingSyncRequests.has(requestId)) {
          console.log('Validation request superseded by newer request');
          break;
        }
        
        const batch = idsToValidate.slice(i, i + BATCH_SIZE);
        
        try {
          const validation = await validateArticles(batch);
          
          // Update cache with batch results
          Object.assign(this.validityCache.results, validation.results);
          this.validityCache.timestamp = Date.now();
          
          // Debounced cache save to avoid frequent localStorage writes
          this.debouncedSaveCache();
        } catch (error) {
          console.error(`Error validating batch ${i} to ${i + batch.length}:`, error);
          
          // CRITICAL FIX: On error, mark ALL IDs in this batch as valid to prevent disappearing content
          batch.forEach(id => {
            this.validityCache.results[id] = true;
          });
        }
      }
      
      // Save cache one final time
      this.saveValidityCache();
    } catch (error) {
      console.error('Error in filterValidArticles:', error);
      // CRITICAL FIX: On any error, return all original articles
      return [...articles];
    } finally {
      this.syncInProgress = false;
      this.pendingSyncRequests.delete(requestId);
    }
    
    // CRITICAL FIX: Changed default behavior to assume articles are valid unless explicitly marked invalid
    // This prevents articles from disappearing if validation fails
    const validArticles = articles.filter(article => 
      this.validityCache.results[article.id] !== false
    );
    
    const invalidCount = articles.length - validArticles.length;
    if (invalidCount > 0) {
      console.warn(`Filtered out ${invalidCount} invalid articles that don't exist in backend`);
    }
    
    return validArticles;
  }
  
  /**
   * Clear the validity cache to force fresh validation
   */
  public clearCache(): void {
    this.validityCache = { timestamp: 0, results: {} };
    if (typeof window !== 'undefined') {
      localStorage.removeItem(LOCAL_STORAGE_KEY);
    }
    console.log('Article validity cache cleared');
  }
} 