'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Header from '../components/Header';

interface SourceData {
  name: string;
  url: string;
  count: number;
  description?: string;
  logo_url?: string;
}

export default function SourcesPage() {
  const [sources, setSources] = useState<SourceData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSources = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/sources');
        
        if (!response.ok) {
          throw new Error('Failed to fetch sources');
        }
        
        const data = await response.json();
        setSources(data.sources || []);
        setError(null);
      } catch (error) {
        console.error('Error fetching sources:', error);
        setError('Failed to load sources. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchSources();
  }, []);

  // Source descriptions (in a real app, these would come from the API)
  const sourceDescriptions: Record<string, string> = {
    'TechCrunch': 'Reporting on the business of technology, startups, venture capital funding, and Silicon Valley.',
    'Wired': 'In-depth coverage of current and future trends in technology and how they are shaping business, entertainment, and culture.',
    'The Verge': 'Covering the intersection of technology, science, art, and culture.',
    'Ars Technica': 'Serving the technologist for more than a decade with deep technical analysis of technology trends.',
    'Hacker News': 'A social news website focusing on computer science and entrepreneurship.',
    'MIT Technology Review': 'Independent media company founded at MIT, covering emerging technologies and their impact.',
    'BBC News': 'Breaking news, features, analysis and debate from the UK and around the world.',
    'CNN': 'Breaking news, latest news and videos from the US and around the world.',
    'The New York Times': 'Breaking news, multimedia, reviews & opinion on Washington, business, sports, movies, travel, books, jobs, education, real estate, cars & more.',
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          News Sources
        </h1>
        
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-white p-6 rounded-lg shadow animate-pulse">
                <div className="flex items-center mb-4">
                  <div className="h-10 w-10 bg-gray-200 rounded-full mr-3"></div>
                  <div className="h-6 bg-gray-200 rounded w-1/3"></div>
                </div>
                <div className="h-4 bg-gray-200 rounded mb-2 w-full"></div>
                <div className="h-4 bg-gray-200 rounded mb-2 w-5/6"></div>
                <div className="h-4 bg-gray-200 rounded w-2/3"></div>
              </div>
            ))}
          </div>
        ) : error ? (
          <div className="bg-red-50 border-l-4 border-red-500 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        ) : sources.length === 0 ? (
          <div className="text-center py-12">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <h3 className="mt-2 text-lg font-medium text-gray-900">No sources found</h3>
            <p className="mt-1 text-sm text-gray-500">
              We couldn't find any sources. Please check back later.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sources.map((source) => (
              <div 
                key={source.name}
                className="bg-white p-6 rounded-lg shadow"
              >
                <div className="flex items-center mb-4">
                  {source.logo_url ? (
                    <img 
                      src={source.logo_url} 
                      alt={`${source.name} logo`} 
                      className="h-10 w-10 object-contain rounded-full mr-3 bg-gray-100 p-1"
                    />
                  ) : (
                    <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center mr-3">
                      <span className="text-blue-800 font-bold">
                        {source.name.charAt(0)}
                      </span>
                    </div>
                  )}
                  <h2 className="text-xl font-semibold text-gray-900">
                    {source.name}
                  </h2>
                </div>
                
                <p className="text-gray-600 mb-4">
                  {source.description || sourceDescriptions[source.name] || `News source for various topics.`}
                </p>
                
                <div className="flex justify-between items-center">
                  <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                    {source.count} articles
                  </span>
                  
                  <div className="flex space-x-2">
                    <Link 
                      href={`/?source=${encodeURIComponent(source.name)}`}
                      className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                      View articles
                    </Link>
                    <a 
                      href={source.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-gray-500 hover:text-gray-700 text-sm font-medium"
                    >
                      Visit website
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
      
      <footer className="bg-white border-t border-gray-200 py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-gray-500 text-sm text-center">
            © {new Date().getFullYear()} Newsletter Aggregator. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
} 