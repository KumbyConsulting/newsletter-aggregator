import { Metadata } from 'next';
import { ArticleProvider } from '../context/ArticleContext';
import { ArticlesContainer } from '../containers/ArticlesContainer';

export const metadata: Metadata = {
  title: 'Articles | News Aggregator',
  description: 'Browse the latest pharmaceutical and healthcare news articles',
};

export default function ArticlesPage() {
  return (
    <ArticleProvider>
      <ArticlesContainer />
    </ArticleProvider>
  );
} 