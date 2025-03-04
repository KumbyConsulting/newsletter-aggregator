'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Header from '../components/Header';

interface TopicData {
  name: string;
  count: number;
  description?: string;
}

export default function TopicsPage() {
  const [topics, setTopics] = useState<TopicData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTopics = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/topics/stats');
        
        if (!response.ok) {
          throw new Error('Failed to fetch topics');
        }
        
        const data = await response.json();
        setTopics(data.topics || []);
        setError(null);
      } catch (error) {
        console.error('Error fetching topics:', error);
        setError('Failed to load topics. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchTopics();
  }, []);

  // Topic descriptions (in a real app, these would come from the API)
  const topicDescriptions: Record<string, string> = {
    Technology: 'Latest news and updates from the tech world, including software, hardware, and digital trends.',
    Business: 'Business news, market trends, entrepreneurship, and corporate developments.',
    Science: 'Scientific discoveries, research breakthroughs, and advancements across various scientific fields.',
    Health: 'Health news, medical research, wellness tips, and healthcare developments.',
    Politics: 'Political news, policy updates, government affairs, and international relations.',
    Entertainment: 'News from the entertainment industry, including movies, music, celebrities, and media.',
    Sports: 'Sports news, game results, athlete updates, and sporting events coverage.',
    Environment: 'Environmental news, climate change updates, conservation efforts, and sustainability initiatives.',
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Browse by Topic
        </h1>
        
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-white p-6 rounded-lg shadow animate-pulse">
                <div className="h-7 bg-gray-200 rounded w-1/3 mb-4"></div>
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
        ) : topics.length === 0 ? (
          <div className="text-center py-12">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <h3 className="mt-2 text-lg font-medium text-gray-900">No topics found</h3>
            <p className="mt-1 text-sm text-gray-500">
              We couldn't find any topics. Please check back later.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {topics.map((topic) => (
              <Link 
                href={`/?topic=${encodeURIComponent(topic.name)}`} 
                key={topic.name}
                className="bg-white p-6 rounded-lg shadow hover:shadow-md transition-shadow"
              >
                <h2 className="text-xl font-semibold text-gray-900 mb-2 flex items-center justify-between">
                  {topic.name}
                  <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                    {topic.count} articles
                  </span>
                </h2>
                <p className="text-gray-600">
                  {topic.description || topicDescriptions[topic.name] || `Articles related to ${topic.name}`}
                </p>
              </Link>
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