import React from 'react';
import ReactMarkdown from 'react-markdown';

interface WeeklyRecapHighlight {
  title: string;
  url: string;
}

interface WeeklyRecapProps {
  recapSummary: string;
  highlights: WeeklyRecapHighlight[];
}

const WeeklyRecap: React.FC<WeeklyRecapProps> = ({ recapSummary, highlights }) => {
  return (
    <div className="bg-white rounded-xl shadow p-6 border border-gray-100" aria-label="Weekly Recap Card">
      <div className="flex items-center mb-4">
        <span className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-green-100 text-green-600 mr-4 text-2xl font-bold" aria-label="Calendar icon">📅</span>
        <h2 className="text-xl font-bold text-gray-800 tracking-tight" aria-label="Weekly Recap Title">Weekly Recap</h2>
      </div>
      <blockquote className="bg-green-50 border-l-4 border-green-400 px-5 py-3 text-base font-medium text-gray-800 mb-4 rounded" aria-label="Recap Summary">
        {recapSummary ? <ReactMarkdown>{recapSummary}</ReactMarkdown> : <span className="text-gray-400 italic">No summary available.</span>}
      </blockquote>
      <div className="mb-2 font-semibold text-gray-700">Highlights:</div>
      <ul className="mb-3 space-y-2" aria-label="Weekly Highlights List">
        {highlights && highlights.length > 0 ? (
          highlights.map((item, i) => (
            <li key={i} className="flex items-start group">
              <span className="w-2 h-2 mt-2 mr-3 rounded-full bg-green-400 flex-shrink-0 group-hover:bg-green-600 transition-colors"></span>
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-green-700 hover:underline font-medium focus:outline-none focus:ring-2 focus:ring-green-400"
                tabIndex={0}
                aria-label={`Highlight: ${item.title}`}
              >
                {item.title || 'Untitled'}
              </a>
            </li>
          ))
        ) : (
          <li className="text-gray-400 italic">No highlights available.</li>
        )}
      </ul>
    </div>
  );
};

export default WeeklyRecap; 