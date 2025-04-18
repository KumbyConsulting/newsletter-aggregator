import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import { Spinner } from './Spinner';
import { ErrorIcon } from './ErrorIcon';
import '../styles/ArticleCard.css';

interface Article {
    id: string;
    title: string;
    description: string;
    source: string;
    sourceUrl: string;
    category?: string;
    pubDate: string;
    analysis?: string;
    analysisMetadata?: {
        is_partial?: boolean;
    };
    sources?: any[];
    imageUrl?: string;
    readTimeMinutes?: number;
}

interface ArticleCardProps {
    article: Article;
    onAnalysisComplete: (article: Article) => void;
}

const ArticleCard: React.FC<ArticleCardProps> = ({ article, onAnalysisComplete }) => {
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [retryCount, setRetryCount] = useState(0);
    const MAX_RETRIES = 2;
    const RETRY_DELAY = 3000; // 3 seconds

    const handleReadClick = () => {
        if (article.sourceUrl) {
            window.open(article.sourceUrl, '_blank', 'noopener,noreferrer');
        } else {
            toast('Source URL not available', {
                icon: '❌'
            });
        }
    };

    const handleShareClick = () => {
        if (navigator.share) {
            navigator.share({
                title: article.title,
                text: article.description,
                url: article.sourceUrl || window.location.href,
            }).catch(error => {
                console.error('Error sharing article:', error);
                fallbackShare();
            });
        } else {
            fallbackShare();
        }
    };

    const fallbackShare = () => {
        // Fallback to copy URL to clipboard
        const shareUrl = article.sourceUrl || window.location.href;
        navigator.clipboard.writeText(shareUrl)
            .then(() => {
                toast('Link copied to clipboard!', {
                    icon: '✅'
                });
            })
            .catch(() => {
                toast('Failed to copy link', {
                    icon: '❌'
                });
            });
    };

    const handleAnalyzeClick = async () => {
        setIsAnalyzing(true);
        setError(null);

        try {
            const response = await fetch(`/api/articles/${article.id}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: AbortSignal.timeout(30000), // 30 second timeout
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Validate response data
            if (!data || !data.analysis) {
                throw new Error('Invalid response format');
            }

            // Handle partial analysis
            if (data.metadata?.is_partial) {
                toast('Only partial analysis available. Some insights may be limited.', {
                    icon: '⚠️'
                });
            }

            // Update the article with analysis data
            onAnalysisComplete({
                ...article,
                analysis: data.analysis,
                analysisMetadata: data.metadata,
                sources: data.sources,
            });

            setRetryCount(0); // Reset retry count on success
            
        } catch (err) {
            const error = err as Error;
            console.error('Analysis error:', error);

            // Handle different error types
            if (error.name === 'AbortError' || error.message.includes('timeout')) {
                if (retryCount < MAX_RETRIES) {
                    setError('Analysis request timed out. Retrying...');
                    setRetryCount(prev => prev + 1);
                    
                    // Retry after delay
                    setTimeout(() => {
                        handleAnalyzeClick();
                    }, RETRY_DELAY);
                    return;
                }
                setError('Analysis timed out. Please try again later.');
            } else if (error.message.includes('404')) {
                setError('Article not found. Please refresh the page.');
            } else if (error.message.includes('429')) {
                setError('Too many requests. Please wait a moment and try again.');
            } else {
                setError('Failed to analyze article. Please try again.');
            }
        } finally {
            if (retryCount >= MAX_RETRIES) {
                setIsAnalyzing(false);
            }
        }
    };

    // Format the date to be more readable
    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now.getTime() - date.getTime());
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays <= 1) return 'Today';
        if (diffDays <= 2) return 'Yesterday';
        if (diffDays <= 7) return `${diffDays} days ago`;
        
        return date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric'
        });
    };

    // Calculate reading time if not provided
    const getReadingTime = () => {
        if (article.readTimeMinutes) return article.readTimeMinutes;
        
        // Estimate reading time based on description length (avg reading speed: ~200 words/minute)
        const wordCount = article.description.split(/\s+/).length;
        return Math.max(1, Math.round(wordCount / 200));
    };

    return (
        <div className="article-card">
            <div className="article-header">
                {article.category && (
                    <span 
                        className="article-category"
                        data-category={article.category}
                    >
                        {article.category}
                    </span>
                )}
                <div className="article-metadata">
                    <span className="article-date">{formatDate(article.pubDate)}</span>
                    <span className="article-source">{article.source}</span>
                    <span className="article-read-time">{getReadingTime()} min read</span>
                </div>
            </div>
            
            {article.imageUrl && (
                <div className="article-image-container">
                    <img 
                        src={article.imageUrl} 
                        alt={article.title} 
                        className="article-image"
                        loading="lazy"
                    />
                </div>
            )}
            
            <h3 className="article-title">{article.title}</h3>
            <p className="article-description">{article.description}</p>
            
            <div className="article-actions">
                <button 
                    className="read-button"
                    onClick={handleReadClick}
                    aria-label="Read article"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M2 12s4-8 10-8 10 8 10 8-4 8-10 8-10-8-10-8z"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                    Read Article
                </button>
                
                <button
                    onClick={handleAnalyzeClick}
                    disabled={isAnalyzing}
                    className={`analyze-button ${isAnalyzing ? 'analyzing' : ''}`}
                    aria-label="Analyze article"
                >
                    {isAnalyzing ? (
                        <>
                            <Spinner size="sm" />
                            {retryCount > 0 ? ` Retry ${retryCount}/${MAX_RETRIES}...` : ' Analyzing...'}
                        </>
                    ) : (
                        <>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                            </svg>
                            Analyze
                        </>
                    )}
                </button>
                
                <button 
                    className="share-button"
                    onClick={handleShareClick}
                    aria-label="Share article"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="18" cy="5" r="3"></circle>
                        <circle cx="6" cy="12" r="3"></circle>
                        <circle cx="18" cy="19" r="3"></circle>
                        <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                        <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                    </svg>
                    Share
                </button>
                
                {error && (
                    <div className="error-message">
                        <ErrorIcon />
                        <span>{error}</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ArticleCard; 