import { NextRequest, NextResponse } from 'next/server';

// Base URL for the backend API - Remove trailing slash to fix URL construction
const API_BASE_URL = process.env.INTERNAL_BACKEND_URL?.replace(/\/$/, '') || 'http://localhost:5000';

// Set a reasonable timeout for backend requests
const FETCH_TIMEOUT = 120000; // 120 seconds

/**
 * Fetch with timeout utility
 */
const fetchWithTimeout = async (url: string, options: RequestInit = {}) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
};

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
    
    // Make the request to the backend with timeout
    const response = await fetchWithTimeout(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      // Add cache control
      next: {
        revalidate: 60, // Cache for 60 seconds
      },
    });
    
    // If the backend returns an error, pass it along with better details
    if (!response.ok) {
      const errorText = await response.text();
      let errorData;
      
      try {
        errorData = JSON.parse(errorText);
      } catch (e) {
        errorData = { message: errorText || `Backend error: ${response.status} ${response.statusText}` };
      }
      
      console.error(`Backend error (${response.status}):`, errorData);
      return NextResponse.json(errorData, { 
        status: response.status,
        headers: Object.fromEntries(response.headers)
      });
    }
    
    // Return the backend response with preserved headers
    const data = await response.json();
    return NextResponse.json(data, {
      headers: Object.fromEntries(response.headers)
    });
  } catch (error) {
    // More detailed error logging and handling
    const isTimeout = error instanceof Error && error.name === 'AbortError';
    const errorMessage = isTimeout 
      ? 'Backend request timed out' 
      : error instanceof Error ? error.message : 'Unknown error';
    
    console.error('Error fetching articles:', { 
      message: errorMessage,
      error: error instanceof Error ? error.stack : String(error)
    });
    
    return NextResponse.json(
      { 
        error: isTimeout ? 'Request timed out' : 'Failed to fetch articles',
        details: errorMessage
      },
      { status: isTimeout ? 504 : 500 }
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
    
    // Make the request to the backend with timeout
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/articles`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    // Return the backend response with preserved headers
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      const data = await response.json();
      return NextResponse.json(data, { 
        status: response.status,
        headers: Object.fromEntries(response.headers)
      });
    } else {
      const text = await response.text();
      return new NextResponse(text, {
        status: response.status,
        headers: {
          'Content-Type': contentType || 'text/plain',
          ...Object.fromEntries(response.headers)
        }
      });
    }
  } catch (error) {
    // More detailed error logging and handling
    const isTimeout = error instanceof Error && error.name === 'AbortError';
    const errorMessage = isTimeout 
      ? 'Backend request timed out' 
      : error instanceof Error ? error.message : 'Unknown error';
    
    console.error('Error posting article:', { 
      message: errorMessage,
      error: error instanceof Error ? error.stack : String(error)
    });
    
    return NextResponse.json(
      { 
        error: isTimeout ? 'Request timed out' : 'Failed to post article',
        details: errorMessage
      },
      { status: isTimeout ? 504 : 500 }
    );
  }
} 