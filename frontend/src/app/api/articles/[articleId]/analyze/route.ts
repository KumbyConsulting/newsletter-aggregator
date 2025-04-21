import { NextRequest, NextResponse } from 'next/server';

// Base URL for the backend API - Remove trailing slash to fix URL construction
const API_BASE_URL = process.env.INTERNAL_BACKEND_URL?.replace(/\/$/, '') || 'http://localhost:5000';

// Set a reasonable timeout for analysis requests
const ANALYSIS_TIMEOUT = 45000; // 45 seconds

/**
 * API route handler for article analysis endpoint
 * Acts as a proxy to the backend API
 */
export async function POST(
  request: NextRequest,
  { params }: { params: { articleId: string } }
) {
  try {
    // Get the article ID from the URL params
    const { articleId } = params;
    
    // Get the request body
    const body = await request.json().catch(() => ({}));
    
    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ANALYSIS_TIMEOUT);
    
    try {
      // Forward the request to the backend
      const response = await fetch(`${API_BASE_URL}/api/articles/${articleId}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: controller.signal
      });
      
      // Clear the timeout
      clearTimeout(timeoutId);
      
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
      
    } catch (fetchError) {
      clearTimeout(timeoutId);
      throw fetchError;
    }
    
  } catch (error) {
    // More detailed error logging and handling
    const isTimeout = error instanceof Error && 
      (error.name === 'AbortError' || error.message.includes('timeout'));
    
    const errorMessage = isTimeout 
      ? 'Analysis request timed out' 
      : error instanceof Error ? error.message : 'Unknown error';
    
    console.error('Error analyzing article:', { 
      message: errorMessage,
      error: error instanceof Error ? error.stack : String(error)
    });
    
    return NextResponse.json(
      { 
        error: isTimeout ? 'Analysis request timed out' : 'Failed to analyze article',
        details: errorMessage
      },
      { status: isTimeout ? 504 : 500 }
    );
  }
} 