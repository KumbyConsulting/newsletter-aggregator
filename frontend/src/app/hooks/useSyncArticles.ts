'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { Article } from '@/types';
import isEqual from 'lodash/isEqual';
import { Article as CoreArticle } from '@/services/api';

/**
 * Hook to ensure articles are synced with backend database
 * Filters out any articles that don't exist in the backend
 * Enhanced with article ID memoization to prevent unnecessary syncs
 * 
 * @param articles Array of articles to sync
 * @returns Object with synced articles and loading state
 */
export default function useSyncArticles(articles: Article[]) {
  const [syncedArticles, setSyncedArticles] = useState<Article[]>(articles);
  const [isSyncing, setIsSyncing] = useState(false);
  const [invalidCount, setInvalidCount] = useState(0);
  const lastSyncedIds = useRef<string[]>([]);
  
  // Get a stable array of just the IDs to reduce unnecessary API calls
  const articleIds = useMemo(() => 
    articles.map(article => article.id).sort(), 
    [articles]
  );
  
  // Check if article IDs have actually changed
  const idsHaveChanged = useMemo(() => {
    // Quick length check first
    if (lastSyncedIds.current.length !== articleIds.length) {
      return true;
    }
    // Then do a deep comparison
    return !isEqual(lastSyncedIds.current, articleIds);
  }, [articleIds]);
  
  // Keep a reference to the current articles for use in effect cleanup
  const articlesRef = useRef(articles);
  useEffect(() => {
    articlesRef.current = articles;
  }, [articles]);

  useEffect(() => {
    console.log('[useSyncArticles] Effect triggered. Article count:', articles?.length);
    console.log('[useSyncArticles] IDs have changed:', idsHaveChanged);
    console.log('[useSyncArticles] Last synced IDs count:', lastSyncedIds.current.length);

    // Skip sync if no articles or if IDs haven't changed from last sync
    if (!articles || articles.length === 0) {
      console.log('[useSyncArticles] Skipping sync: No articles.');
      setSyncedArticles([]);
      return;
    }
    
    // Skip sync if article IDs haven't changed
    if (!idsHaveChanged && lastSyncedIds.current.length > 0) {
      console.log('[useSyncArticles] Skipping sync: IDs have not changed.');
      return;
    }
    
    // Track if component is still mounted
    let isMounted = true;
    // Use a sync timeout to prevent rapid successive syncs
    let syncTimeoutId: NodeJS.Timeout;

    const syncArticles = async () => {
      // Don't set syncing state immediately to prevent UI flicker for fast syncs
      const syncingStateTimeoutId = setTimeout(() => {
        if (isMounted) {
          console.log('[useSyncArticles] Setting isSyncing to true (after delay)');
          setIsSyncing(true);
        }
      }, 150); // Only show loading state if sync takes longer than 150ms
      
      try {
        // Dynamically import to avoid SSR issues
        const { SyncService } = await import('../services/syncService');
        const syncService = SyncService.getInstance();
        
        // Filter articles to only include valid ones
        console.log('[useSyncArticles] Calling syncService.filterValidArticles...');
        
        // Try to sync articles, but with fallback to original articles if anything fails
        let validArticles;
        try {
          validArticles = await syncService.filterValidArticles(articlesRef.current);
        } catch (syncError) {
          console.error('Error in syncService.filterValidArticles:', syncError);
          // CRITICAL FIX: On any error, keep all original articles instead of filtering
          validArticles = [...articlesRef.current];
        }
        
        // Update state if component is still mounted
        if (isMounted) {
          const newInvalidCount = articlesRef.current.length - validArticles.length;
          setInvalidCount(newInvalidCount);
          setSyncedArticles(validArticles);
          lastSyncedIds.current = articleIds; // Update last synced IDs
        }
      } catch (error) {
        console.error('Error syncing articles:', error);
        // On error, fall back to original articles
        if (isMounted) {
          setSyncedArticles(articlesRef.current);
        }
      } finally {
        // Clear the syncing state timeout
        clearTimeout(syncingStateTimeoutId);
        if (isMounted) {
          setIsSyncing(false);
        }
      }
    };

    // Add a small delay to avoid rapid consecutive API calls
    console.log('[useSyncArticles] Scheduling syncArticles call (300ms delay)...');
    syncTimeoutId = setTimeout(syncArticles, 300);

    // Cleanup function
    return () => {
      isMounted = false;
      clearTimeout(syncTimeoutId);
    };
  }, [articles, idsHaveChanged, articleIds]);

  return {
    articles: syncedArticles,
    isSyncing,
    invalidCount,
    hasSyncedArticles: !isSyncing && syncedArticles.length > 0
  };
} 