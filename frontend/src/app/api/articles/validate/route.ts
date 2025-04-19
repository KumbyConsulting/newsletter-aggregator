import { NextRequest, NextResponse } from 'next/server';

// Base URL for the backend API - Remove trailing slash to fix URL construction
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') || 'http://localhost:5000';

/**
 * API route handler for validating article IDs
 * Acts as a proxy to the backend API
 */
export async function POST(request: NextRequest) {
  try {
    // Extract article IDs from the request body
    const body = await request.json();
    const articleIds = body.article_ids || [];
    
    if (!articleIds.length) {
      return NextResponse.json({
        results: {},
        count: {
          total: 0,
          valid: 0,
          invalid: 0
        }
      });
    }
    
    // Forward the request to the backend
    const response = await fetch(`${API_BASE_URL}/api/articles/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ article_ids: articleIds }),
      // Add cache control
      next: {
        revalidate: 60, // Cache for 60 seconds
      },
    });
    
    // If the backend returns an error, pass it along
    if (!response.ok) {
      // For validation endpoints, return all articles as valid if backend fails
      // This prevents disappearing articles in case of backend issues
      const validResults: Record<string, boolean> = {};
      articleIds.forEach((id: string) => {
        validResults[id] = true;
      });
      
      return NextResponse.json({
        results: validResults,
        count: {
          total: articleIds.length,
          valid: articleIds.length,
          invalid: 0
        }
      });
    }
    
    // Return the backend response
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error validating articles:', error);
    
    // In case of error, consider all articles valid
    // This prevents articles from disappearing unnecessarily
    const articleIds = [];
    try {
      const body = await request.json();
      articleIds.push(...(body.article_ids || []));
    } catch (e) {
      // If we can't parse the body, return an empty result
    }
    
    const validResults: Record<string, boolean> = {};
    articleIds.forEach((id: string) => {
      validResults[id] = true;
    });
    
    return NextResponse.json({
      results: validResults,
      count: {
        total: articleIds.length,
        valid: articleIds.length,
        invalid: 0
      }
    });
  }
} 