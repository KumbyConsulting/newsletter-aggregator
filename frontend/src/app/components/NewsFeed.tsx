import React from 'react';

interface Article {
  title: string;
  summary: string;
  date: string;
  source: string;
}

interface NewsFeedProps {
  articles: Article[];
}

const NewsFeed: React.FC<NewsFeedProps> = ({ articles }) => {
  return (
    <div>
      <h2>Latest News</h2>
      <ul>
        {articles.map((article, i) => (
          <li key={i}>
            <h3>{article.title}</h3>
            <p>{article.summary}</p>
            <span>{article.date} - {article.source}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default NewsFeed; 