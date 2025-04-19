import { NextResponse } from 'next/server';

// Base URL for the backend API - Remove trailing slash to fix URL construction
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') || 'http://localhost:5000';

/**
 * API route handler for topics endpoint
 * Acts as a proxy to the backend API with built-in caching
 */
export async function GET() {
  try {
    // Make the request to the backend
    const response = await fetch(`${API_BASE_URL}/api/topics`, {
      headers: {
        'Content-Type': 'application/json',
      },
      // Add stronger caching for topics since they rarely change
      next: {
        revalidate: 3600, // Cache for 1 hour
      },
    });
    
    // If the backend returns an error, pass it along
    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(error, { status: response.status });
    }
    
    // Return the backend response
    const data = await response.json();
    
    // Set cache control headers for the client
    return NextResponse.json(data, {
      headers: {
        'Cache-Control': 'public, max-age=3600, s-maxage=3600',
      },
    });
  } catch (error) {
    console.error('Error fetching topics:', error);
    return NextResponse.json(
      { error: 'Failed to fetch topics' },
      { status: 500 }
    );
  }
} 