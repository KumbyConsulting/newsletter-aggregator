export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h1 className="text-3xl font-bold text-gray-900">
              About Newsletter Aggregator
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-gray-500">
              Learn more about our platform and mission.
            </p>
          </div>
          
          <div className="border-t border-gray-200 px-4 py-5 sm:px-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Our Mission</h2>
            <p className="text-gray-700 mb-6">
              Newsletter Aggregator is designed to help you stay informed without the overwhelm. 
              We collect articles from various sources across the web, categorize them by topic, 
              and present them in a clean, easy-to-navigate interface.
            </p>
            
            <h2 className="text-xl font-semibold text-gray-900 mb-4">How It Works</h2>
            <p className="text-gray-700 mb-4">
              Our platform regularly scans RSS feeds from trusted sources to gather the latest articles.
              We use natural language processing to categorize content by topic and extract key information.
            </p>
            <p className="text-gray-700 mb-6">
              You can browse articles by topic, search for specific content, and even find similar articles
              related to ones you're interested in.
            </p>
            
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Features</h2>
            <ul className="list-disc pl-5 text-gray-700 mb-6 space-y-2">
              <li>Automatic aggregation from multiple sources</li>
              <li>Topic-based categorization</li>
              <li>Full-text search</li>
              <li>Similar article recommendations</li>
              <li>Clean, responsive interface</li>
              <li>Regular updates with fresh content</li>
            </ul>
            
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Technology</h2>
            <p className="text-gray-700 mb-6">
              Newsletter Aggregator is built with modern web technologies:
            </p>
            <ul className="list-disc pl-5 text-gray-700 mb-6 space-y-2">
              <li>Next.js for the frontend</li>
              <li>Flask for the backend API</li>
              <li>SQLite for data storage</li>
              <li>Natural language processing for content analysis</li>
            </ul>
            
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Contact</h2>
            <p className="text-gray-700">
              Have questions or suggestions? Feel free to reach out to us at{' '}
              <a href="mailto:contact@newsletter-aggregator.com" className="text-blue-600 hover:underline">
                contact@newsletter-aggregator.com
              </a>
            </p>
          </div>
        </div>
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