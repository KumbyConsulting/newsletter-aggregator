import { NextRequest, NextResponse } from 'next/server';

// Base URL for the backend API - Remove trailing slash to fix URL construction
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') || 'http://localhost:5000';

/**
 * API route handler for articles endpoint
 * Acts as a proxy to the backend API
 */
export async function GET(request: NextRequest) {
  try {
    // Extract search params from the request
    const searchParams = request.nextUrl.searchParams;
    
    // Forward the search params to the backend
    const url = `${API_BASE_URL}/api/articles?${searchParams.toString()}`;
    
    // Make the request to the backend
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      // Add cache control
      next: {
        revalidate: 60, // Cache for 60 seconds
      },
    });
    
    // If the backend returns an error, pass it along
    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(error, { status: response.status });
    }
    
    // Return the backend response
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching articles:', error);
    return NextResponse.json(
      { error: 'Failed to fetch articles' },
      { status: 500 }
    );
  }
}

/**
 * API route handler for individual article
 */
export async function POST(request: NextRequest) {
  try {
    // Get the request body
    const body = await request.json();
    
    // Make the request to the backend
    const response = await fetch(`${API_BASE_URL}/api/articles`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    // Return the backend response
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('Error posting article:', error);
    return NextResponse.json(
      { error: 'Failed to post article' },
      { status: 500 }
    );
  }
} 