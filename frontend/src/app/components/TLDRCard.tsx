import React from 'react';
import ReactMarkdown from 'react-markdown';

interface TLDRSource {
  title: string;
  url: string;
  topic?: string;
  source?: string;
  pub_date?: string;
  image_url?: string;
  description?: string;
  summary?: string;
  reading_time?: number;
  has_full_content?: boolean;
}

interface TLDRCardProps {
  summary: string;
  highlights: TLDRSource[];
  sources: TLDRSource[];
  updatedAgo?: string; // Optional: pass in 'Updated X minutes ago' if available
}

const highlightIcons = ['★', '🔥', '📈', '📰'];

// Subcomponent: HighlightItem
const HighlightItem: React.FC<{ item: TLDRSource; index: number }> = ({ item, index }) => (
  <li
    key={index}
    className={`flex flex-col sm:flex-row sm:items-start tldr-highlight-item ${index === 0 ? 'font-bold text-blue-800' : ''} bg-gray-50 rounded-lg p-3 shadow-sm transition-colors hover:bg-blue-50 focus-within:ring-2 focus-within:ring-blue-300`}
    tabIndex={0}
    aria-label={`Highlight ${index + 1}: ${item.title}`}
  >
              {/* Image if present */}
              {item.image_url && (
                <div className="flex-shrink-0 mr-4 mb-2 sm:mb-0">
        <img src={item.image_url} alt={item.title || 'Highlight image'} className="w-24 h-16 object-cover rounded-md border" />
                </div>
              )}
              <div className="flex-1">
                <div className="flex items-center flex-wrap mb-1">
        <span className="mr-2 text-lg" aria-hidden="true">{highlightIcons[index] || '📰'}</span>
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
          className="highlight-link text-blue-700 hover:underline text-base font-semibold focus:outline-none focus:ring-2 focus:ring-blue-400"
          tabIndex={0}
                  >
          {item.title || 'Untitled'}
                  </a>
        {index === 0 && (
                    <span className="ml-2 px-2 py-0.5 bg-yellow-200 text-yellow-800 text-xs rounded-full font-semibold">Top Story</span>
                  )}
                  {item.reading_time && (
                    <span className="ml-3 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full flex items-center">
                      ⏱ {item.reading_time} min read
                    </span>
                  )}
                  {item.has_full_content && (
                    <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">Full Content</span>
                  )}
                </div>
                <div className="flex flex-wrap items-center mb-1 space-x-2">
                  {item.topic && (
                    <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded-full text-xs font-medium mr-1">{item.topic}</span>
                  )}
                  {item.source && (
                    <span className="bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full text-xs font-medium mr-1">{item.source}</span>
                  )}
                  {item.pub_date && (
                    <span className="text-xs text-gray-400">{item.pub_date}</span>
                  )}
                </div>
                {item.description && (
                  <div className="text-gray-600 text-sm mt-1 line-clamp-2">{item.description}</div>
                )}
                {item.summary && (
                  <div className="text-gray-800 text-xs italic mt-1">Summary: {item.summary}</div>
                )}
              </div>
            </li>
);

// Subcomponent: SourcePill
const SourcePill: React.FC<{ src: TLDRSource }> = ({ src }) => (
            <a
              href={src.url}
              target="_blank"
              rel="noopener noreferrer"
    className="source-pill bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full mr-2 mb-1 hover:bg-blue-200 transition-colors flex items-center focus:outline-none focus:ring-2 focus:ring-blue-400"
    tabIndex={0}
    aria-label={`Source: ${src.title}`}
            >
    {src.title || 'Untitled'}
              {src.source && (
                <span className="ml-1 text-gray-500">({src.source})</span>
              )}
              {src.topic && (
                <span className="ml-1 text-green-600">[{src.topic}]</span>
              )}
              {src.reading_time && (
                <span className="ml-2 text-xs text-blue-700">⏱ {src.reading_time}m</span>
              )}
              {src.has_full_content && (
                <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">Full</span>
              )}
            </a>
);

const TLDRCard: React.FC<TLDRCardProps> = ({ summary, highlights, sources, updatedAgo }) => {
  return (
    <div className="tldr-gradient-border rounded-xl shadow p-0 border bg-white" aria-label="TLDR summary card">
      <div className="flex justify-between items-center px-6 pt-5 pb-2">
        <span className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-blue-100 text-blue-600 mr-3 text-2xl font-bold tldr-badge" aria-label="TLDR badge">
          ⚡ TL;DR
        </span>
        <span className="text-xs text-gray-400 tldr-updated" aria-label="Last updated">{updatedAgo || "Today"}</span>
      </div>
      <blockquote className="tldr-summary-block bg-blue-50 border-l-4 border-blue-400 px-6 py-4 text-lg font-semibold text-gray-800 flex items-start" aria-label="Summary">
        <span className="mr-2 text-blue-400 text-xl" aria-hidden="true">"</span>
        <span className="flex-1">
          {summary ? (
            <ReactMarkdown
              className="tldr-markdown"
              components={{
                p: ({ node, ...props }) => <p {...props} className="mb-2 last:mb-0" />,
                ul: ({ node, ...props }) => <ul {...props} className="list-disc ml-6 mb-2" />,
                ol: ({ node, ...props }) => <ol {...props} className="list-decimal ml-6 mb-2" />,
                li: ({ node, ...props }) => <li {...props} className="mb-1" />,
              }}
            >
              {summary}
            </ReactMarkdown>
          ) : (
            <span className="text-gray-400 italic">No summary available.</span>
          )}
        </span>
      </blockquote>
      <div className="px-6 pt-3 pb-2">
        <div className="font-semibold text-gray-700 mb-2 tldr-highlights-title">Today's Highlights:</div>
        <ul className="mb-2 space-y-3" aria-label="Highlights list">
          {highlights && highlights.length > 0 ? (
            highlights.map((item, i) => <HighlightItem item={item} index={i} key={i} />)
          ) : (
            <li className="text-gray-400 italic">No highlights available.</li>
          )}
        </ul>
        <hr className="my-3 border-gray-200" />
        <div className="text-xs text-gray-500 mt-3 flex items-center flex-wrap tldr-sources-row" aria-label="Sources row">
          <span className="font-semibold mr-2 sources-label">Sources:</span>
          {sources && sources.length > 0 ? (
            sources.map((src, i) => <SourcePill src={src} key={i} />)
          ) : (
            <span className="text-gray-400 italic">No sources available.</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default TLDRCard; 