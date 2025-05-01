import { NextRequest, NextResponse } from 'next/server';

// Base URL for the backend API - Remove trailing slash to fix URL construction
const API_BASE_URL = process.env.INTERNAL_BACKEND_URL?.replace(/\/$/, '') || 'http://localhost:5000';

// Set a shorter timeout for status checks
const STATUS_TIMEOUT = 120000; // 120 seconds

/**
 * API route handler for update status endpoint
 * Acts as a proxy to the backend API
 */
export async function GET(request: NextRequest) {
  try {
    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), STATUS_TIMEOUT);
    
    try {
      // Forward the request to the backend
      const response = await fetch(`${API_BASE_URL}/api/update/status`, {
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal,
        // Add cache control
        next: {
          revalidate: 5, // Cache for 5 seconds
        },
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
      ? 'Status check timed out' 
      : error instanceof Error ? error.message : 'Unknown error';
    
    console.error('Error checking update status:', { 
      message: errorMessage,
      error: error instanceof Error ? error.stack : String(error)
    });
    
    // For status checks, return a fallback status on error
    return NextResponse.json({
      in_progress: false,
      last_update: null,
      status: 'unknown',
      progress: 0,
      message: 'Unable to connect to server',
      error: errorMessage,
      sources_processed: 0,
      total_sources: 0,
      articles_found: 0,
      estimated_completion_time: null,
      can_be_cancelled: false
    }, { 
      status: isTimeout ? 504 : 500 
    });
  }
} 