'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  Typography,
  Button,
  Space,
  Modal,
  Tooltip,
  Tag,
  message,
  Spin,
  Empty,
  List,
  Avatar,
  Divider,
  Alert,
  Select,
  DatePicker,
  Input
} from 'antd';
import {
  CalendarOutlined,
  LinkOutlined,
  ShareAltOutlined,
  LikeOutlined,
  UserOutlined,
  ClockCircleOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
  TagOutlined,
  SearchOutlined,
  ArrowRightOutlined,
  BookOutlined,
  FileTextOutlined,
  InfoCircleOutlined,
  BulbOutlined,
  FilterOutlined,
  SortAscendingOutlined
} from '@ant-design/icons';
import Image from 'next/image';
import DOMPurify from 'isomorphic-dompurify';
import { formatDistanceToNow } from 'date-fns';
import { Article } from '@/types';
import { marked } from 'marked';

const { Title, Text, Paragraph } = Typography;

interface ArticleCardProps {
  article: Article;
  className?: string;
}

// Add a new component to inject the styles
const ArticleCardStyles = () => {
  useEffect(() => {
    // Only inject the styles once
    if (!document.getElementById('article-card-styles')) {
      const styleElement = document.createElement('style');
      styleElement.id = 'article-card-styles';
      styleElement.innerHTML = styles;
      document.head.appendChild(styleElement);
    }
    
    // Clean up on unmount
    return () => {
      const styleElement = document.getElementById('article-card-styles');
      if (styleElement) {
        styleElement.remove();
      }
    };
  }, []);
  
  return null;
};

export const ArticleCard: React.FC<ArticleCardProps> = ({ article, className }) => {
  const { metadata } = article;
  const {
    title,
    description,
    link,
    pub_date,
    topic,
    source,
    image_url,
    has_full_content,
    reading_time,
    is_recent,
    summary
  } = metadata;

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [analysisModalVisible, setAnalysisModalVisible] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formattedDate, setFormattedDate] = useState<string>('');

  // Format the publication date on the client side only
  useEffect(() => {
    try {
      if (!pub_date) {
        setFormattedDate('No date');
        return;
      }
      
      // Handle timezone-aware dates
      const parsedDate = pub_date.endsWith('Z') || pub_date.includes('+') || pub_date.includes('-') ?
        new Date(pub_date) :
        new Date(pub_date + 'Z'); // Assume UTC if no timezone specified
      
      // Validate the date before formatting
      if (isNaN(parsedDate.getTime())) {
        console.error("Invalid date detected:", pub_date);
        setFormattedDate('Invalid Date');
        return;
      }
        
      setFormattedDate(formatDistanceToNow(parsedDate, { addSuffix: true }));
    } catch (e) {
      console.error("Error formatting date:", pub_date, e);
      setFormattedDate('Invalid Date');
    }
  }, [pub_date]);

  // Sanitize HTML content for modal display only if it's likely HTML
  const contentForModal = (has_full_content && description?.includes('<'))
    ? DOMPurify.sanitize(description)
    : description;

  // Create plain text snippet for card display
  const snippetForCard = description?.includes('<')
    ? DOMPurify.sanitize(description, { USE_PROFILES: { html: false } })
    : description;

  // Share function
  const handleShare = async () => {
    const shareData = {
      title: title,
      text: `Check out this article: ${title}`,
      url: link,
    };
    try {
      if (navigator.share) {
        await navigator.share(shareData);
      } else {
        // Fallback for browsers that don't support navigator.share
        navigator.clipboard.writeText(link);
        message.success('Article link copied to clipboard!');
      }
    } catch (error) {
      console.error('Error sharing:', error);
      message.error('Could not share article.');
    }
  };

  const showModal = () => {
    setIsModalVisible(true);
  };

  const handleOk = () => {
    setIsModalVisible(false);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
  };

  const handleAnalyzeClick = async () => {
    try {
      setLoading(true);
      setError(null);
      setAnalysisModalVisible(true);

      // Add timeout to the fetch request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 45000); // 45 second timeout

      try {
        const response = await fetch(`/api/articles/${article.id}/analyze`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({}),
          signal: controller.signal
        });

        clearTimeout(timeoutId); // Clear the timeout

        // Handle partial content response (206)
        if (response.status === 206) {
          const data = await response.json();
          setAnalysis({
            analysis: data.analysis || 'Partial analysis available. The analysis system was unable to complete the full analysis.',
            metadata: {
              confidence_score: data.metadata?.confidence_score || 0.5,
              context_articles_count: data.metadata?.context_articles_count || 0,
              generated_at: new Date().toISOString(),
              analysis_type: data.metadata?.analysis_type || 'general',
              is_partial: true
            },
            sources: data.sources || []
          });
          
          // Show warning message for partial analysis
          message.warning('Only partial analysis is available. Some insights may be limited.');
          return;
        }
        
        // Handle other error responses
        if (!response.ok) {
          const errorData = await response.json().catch(() => null);
          
          // Handle specific status codes
          if (response.status === 429) {
            throw new Error('Rate limit exceeded. Please try again later.');
          } else if (response.status === 404) {
            throw new Error('Article not found. It may have been removed from the database.');
          } else if (response.status === 503) {
            throw new Error('AI service is currently unavailable. Please try again later.');
          }
          
          throw new Error(errorData?.description || `Failed to generate analysis (${response.status})`);
        }

        const data = await response.json();
        
        // Validate the response data - Check the actual structure
        if (!data || typeof data !== 'object' || !data.result) {
          throw new Error('Invalid analysis response format');
        }

        // Extract analysis, metadata, and sources from the 'result' field
        const analysisContent = data.result.text || 'Analysis content unavailable';
        const analysisMetadata = {
            ...data.result.metadata || {},
            confidence_score: data.result.confidence || 0,
            context_articles_count: data.result.metadata?.context_articles_count || 0,
            generated_at: data.result.metadata?.generated_at || new Date().toISOString(),
            analysis_type: data.analysis_type || 'simplified',
            is_partial: !data.result.text
        };
        const analysisSources = data.result.sources || [];

        // Ensure sources array exists and format if needed (assuming backend sends structured sources)
        // No change needed here if backend already sends structured sources inside data.result.sources

        // Set the state with the correct structure expected by renderAnalysisContent
        setAnalysis({
          analysis: analysisContent,
          metadata: analysisMetadata,
          sources: analysisSources
        });

      } catch (fetchError: unknown) {
        clearTimeout(timeoutId);
        
        // Handle abort error
        if (fetchError instanceof Error && fetchError.name === 'AbortError') {
          throw new Error('Analysis request timed out. The server took too long to respond.');
        }
        
        throw fetchError;
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate analysis');
      message.error('Failed to generate analysis. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const renderAnalysisContent = () => {
    if (loading) {
      return (
        <div className="analysis-loading-container" style={{ padding: '32px', textAlign: 'center' }}>
          <Spin size="large" />
          <p className="mt-4" style={{ marginTop: '16px', fontSize: '16px' }}>Generating comprehensive analysis...</p>
          <Text type="secondary" style={{ fontSize: '14px' }}>
            This may take a few moments as we analyze the article and related content
          </Text>
        </div>
      );
    }

    if (error) {
      return (
        <div className="analysis-error-container" style={{ padding: '24px' }}>
          <Alert
            type="error"
            message="Analysis Error"
            description={error}
            action={
              <Button type="primary" onClick={handleAnalyzeClick}>
                Retry Analysis
              </Button>
            }
          />
        </div>
      );
    }

    if (!analysis) {
      return null;
    }

    return (
      <div className="analysis-content-wrapper">
        {/* Metadata Section */}
        <div className="analysis-content">
          <div className="analysis-metadata">
            <Space wrap size={[8, 8]}>
              <Tag color="blue">
                Confidence Score: {((analysis.metadata?.confidence_score || 0) * 100).toFixed(0)}%
              </Tag>
              <Tag color="green">
                Context Articles: {analysis.metadata?.context_articles_count || 0}
              </Tag>
              <Tag color="purple">
                Generated: {new Date(analysis.metadata?.generated_at || Date.now()).toLocaleString()}
              </Tag>
              {analysis.metadata?.is_partial && (
                <Tag color="orange">Partial Analysis</Tag>
              )}
            </Space>
          </div>
          
          <Divider style={{ margin: '12px 0', borderColor: '#eee' }} />
          
          {/* Content Section */}
          <div 
            className="markdown-content" 
            dangerouslySetInnerHTML={{ 
              __html: marked.parse(analysis.analysis || 'No analysis content available') as string 
            }} 
          />
        </div>
        
        {/* Sources Section */}
        {Array.isArray(analysis.sources) && analysis.sources.length > 0 && (
          <div className="analysis-sources-container">
            <Divider orientation="left" style={{ margin: '8px 16px', fontWeight: 'bold' }}>
              Sources ({analysis.sources.length})
            </Divider>
            <ul className="sources-list">
              {analysis.sources.map((source: any, index: number) => (
                <li key={index}>
                  {source.link ? (
                    <a href={source.link} target="_blank" rel="noopener noreferrer" className="source-link">
                      {source.title || 'Untitled Source'}
                    </a>
                  ) : (
                    <Text strong>{source.title || 'Untitled Source'}</Text>
                  )}
                  {(source.source || source.date) && (
                    <Text type="secondary" className="source-metadata">
                      {' - '}
                      {source.source && <span>{source.source}</span>}
                      {source.source && source.date && ', '}
                      {source.date && (
                        <span>
                          {(() => {
                            try {
                              // Validate the date before formatting it
                              const dateObj = new Date(source.date);
                              if (isNaN(dateObj.getTime())) {
                                return 'Invalid date';
                              }
                              return formatDistanceToNow(dateObj, { addSuffix: true });
                            } catch (e) {
                              console.error("Error formatting date:", source.date, e);
                              return 'Invalid date';
                            }
                          })()}
                        </span>
                      )}
                    </Text>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      <ArticleCardStyles />
      <Card 
        hoverable
        className={`article-card ${className || ''}`}
        cover={image_url && (
          <div className="article-image-container cursor-pointer" onClick={showModal}>
            <Image
              src={image_url}
              alt={title}
              layout="fill"
              objectFit="cover"
              className="article-image"
            />
          </div>
        )}
        actions={[
          <Tooltip title="Generate comprehensive analysis">
            <Button 
              type="text" 
              icon={<BulbOutlined />} 
              onClick={handleAnalyzeClick}
              loading={loading}
            >
              Analyze
            </Button>
          </Tooltip>
        ]}
      >
        <Space direction="vertical" size="small" className="w-full">
          {/* Topic and Badges */}
          <Space wrap>
            <Tag color="blue">{topic}</Tag>
            {is_recent && <Tag color="green">New</Tag>}
            {has_full_content && (
              <Tooltip title="Full content available in details">
                <Tag icon={<FileTextOutlined />} color="purple">Full</Tag>
              </Tooltip>
            )}
            {summary && (
              <Tooltip title="AI Summary available in details">
                <Tag icon={<InfoCircleOutlined />} color="gold">Summary</Tag>
              </Tooltip>
            )}
          </Space>

          {/* Title */}
          <Title
            level={4}
            className="article-title cursor-pointer hover:text-blue-600"
            ellipsis={{ rows: 2 }}
            onClick={showModal}
            title={title}
          >
            <a href={link} target="_blank" rel="noopener noreferrer">
              {title}
            </a>
          </Title>

          {/* Description */}
          <Paragraph ellipsis={{ rows: 3 }} className="article-description">
            {snippetForCard}
          </Paragraph>

          {/* Metadata Footer */}
          <Space split="•" className="text-gray-500 text-sm" style={{ flexWrap: 'wrap' }}>
            <Space>
              <BookOutlined />
              <Text>{source}</Text>
            </Space>
            <Text>{formattedDate}</Text>
            {reading_time && (
              <Tooltip title="Estimated reading time">
                <Space>
                  <ClockCircleOutlined />
                  <span>{reading_time} min</span>
                </Space>
              </Tooltip>
            )}
          </Space>

          {/* Action Buttons */}
          <Space className="mt-3 justify-between items-center w-full">
            <Button
              type="primary"
              size="small"
              onClick={showModal}
              icon={<EyeOutlined />}
            >
              View Details
            </Button>
            <Space size="small">
              <Tooltip title="Open original article">
                <Button
                  type="text"
                  size="small"
                  href={link}
                  target="_blank"
                  rel="noopener noreferrer"
                  icon={<LinkOutlined />}
                  aria-label="Open original article"
                />
              </Tooltip>
              <Tooltip title="Share Article">
                <Button
                  type="text"
                  size="small"
                  icon={<ShareAltOutlined />}
                  onClick={handleShare}
                  aria-label="Share article"
                />
              </Tooltip>
            </Space>
          </Space>
        </Space>
      </Card>

      <Modal
        title={<Title level={4} style={{marginBottom: 0}} ellipsis={{rows: 2}}>{title}</Title>}
        open={isModalVisible}
        onOk={handleOk}
        onCancel={handleCancel}
        width={800}
        destroyOnClose
        className="article-modal"
        footer={[
          <Button key="share" icon={<ShareAltOutlined />} onClick={handleShare}>
            Share
          </Button>,
          <Button key="link" type="primary" href={link} target="_blank" rel="noopener noreferrer" icon={<LinkOutlined />}>
            View Original Article
          </Button>,
          <Button key="close" onClick={handleCancel}>
            Close
          </Button>,
        ]}
        styles={{
          body: {
            maxHeight: '70vh',
            overflowY: 'auto'
          }
        }}
      >
        {/* Header Info Section */}
        <div className="article-modal-header">
          {/* Tags and Badges */}
          <Space wrap className="mb-3">
            <Tag color="blue">{topic}</Tag>
            {is_recent && <Tag color="green">New</Tag>}
            {has_full_content && (
              <Tag icon={<FileTextOutlined />} color="purple">Full Content</Tag>
            )}
            {reading_time && (
              <Tag icon={<ClockCircleOutlined />}>{reading_time} min read</Tag>
            )}
          </Space>
          
          {/* Metadata */}
          <Space direction="vertical" size="small" className="w-full mb-2">
            <Space size="small" className="text-gray-600 text-xs" wrap>
              <Space><BookOutlined /> <Text strong>{source}</Text></Space>
              •
              <Space><CalendarOutlined /> <Text>{formattedDate}</Text></Space>
              {article.metadata.relevance_score !== undefined && (
                <>• <Space><SearchOutlined /> <Text>Relevance: {(article.metadata.relevance_score * 100).toFixed(0)}%</Text></Space></>
              )}
            </Space>
            
            {/* Source Link */}
            <Text className="text-xs">
              <LinkOutlined className="mr-1" />
              <a href={link} target="_blank" rel="noopener noreferrer">{link}</a>
            </Text>
            
            {/* ID (for reference) */}
            <Text type="secondary" className="text-xs">ID: {article.id}</Text>
          </Space>
        </div>
        <Divider />

        {/* Image if available */}
        {image_url && (
          <div className="article-modal-image mb-4">
            <img 
              src={image_url} 
              alt={title} 
              style={{ 
                maxWidth: '100%', 
                maxHeight: '300px', 
                objectFit: 'contain',
                display: 'block',
                margin: '0 auto'
              }} 
            />
          </div>
        )}

        {/* AI Summary Section */}
        {summary && (
          <>
            <Title level={5}><InfoCircleOutlined style={{ marginRight: 8, color: '#faad14' }} /> AI Generated Summary</Title>
            <Paragraph className="bg-yellow-50 p-3 rounded border border-yellow-200 text-sm" style={{ color: 'rgba(0, 0, 0, 0.85)' }}>
              {summary}
            </Paragraph>
            <Divider />
          </>
        )}

        {/* Article Content Section */}
        <Title level={5}>{has_full_content ? 'Full Article Content' : 'Article Snippet'}</Title>
        <div
          className="prose max-w-none article-modal-content article-snippet"
          dangerouslySetInnerHTML={{ __html: contentForModal }}
          style={{ color: 'rgba(0, 0, 0, 0.85)' }}
        />
        {!has_full_content && description && (
            <Paragraph type="secondary" className="mt-4 italic text-xs" style={{ color: 'rgba(0, 0, 0, 0.65)' }}>
                Note: This is only a snippet. Visit the original article for the full content.
            </Paragraph>
        )}
        {!description && (
             <Paragraph type="secondary" className="mt-4 italic text-xs" style={{ color: 'rgba(0, 0, 0, 0.65)' }}>
                No content available for this article.
            </Paragraph>
        )}
      </Modal>

      {/* Analysis Modal */}
      <Modal
        title={
          <div className="analysis-modal-title">
            <Space align="center">
              <BulbOutlined style={{ color: '#1890ff', fontSize: '20px' }} />
              <Title level={4} style={{ margin: 0 }}>
                Comprehensive Analysis: {title.length > 40 ? title.substring(0, 40) + '...' : title}
              </Title>
            </Space>
          </div>
        }
        open={analysisModalVisible}
        onCancel={() => setAnalysisModalVisible(false)}
        width={800}
        footer={
          <Button type="primary" onClick={() => setAnalysisModalVisible(false)}>
            Close
          </Button>
        }
        styles={{
          body: {
            maxHeight: '70vh',
            overflowY: 'auto',
            padding: '0',
            backgroundColor: '#f9fafb'
          },
          header: {
            padding: '16px 24px',
            borderBottom: '1px solid #f0f0f0'
          },
          footer: {
            padding: '12px 24px',
            borderTop: '1px solid #f0f0f0'
          },
          content: {
            borderRadius: '12px',
            overflow: 'hidden'
          }
        }}
        className="analysis-modal"
        destroyOnClose={true}
      >
        {renderAnalysisContent()}
      </Modal>
    </>
  );
};

// Loading skeleton for the article card
export const ArticleCardSkeleton: React.FC = () => {
  return (
    <Card loading className="article-card-skeleton" />
  );
};

// Add styles to your global CSS or a separate module
const styles = `
.article-card {
  height: 100%;
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
  border: 1px solid var(--border-color, #e8e8e8);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  background: #ffffff;
  border-radius: 12px;
  overflow: hidden;
  padding: 0;
}

.article-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
  border-color: var(--primary-color, #1890ff);
}

.article-image-container {
  position: relative;
  width: 100%;
  height: 220px;
  overflow: hidden;
  background-color: #f5f5f5;
}

.article-card .ant-card-body {
  padding: 20px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.article-title {
  margin-top: 8px !important;
  margin-bottom: 16px !important;
  font-size: 1.25rem !important;
  line-height: 1.5 !important;
  font-weight: 600 !important;
  color: var(--heading-color, #262626) !important;
  transition: color 0.3s ease;
  letter-spacing: -0.01em;
}

.article-description {
  color: var(--text-color, rgba(0, 0, 0, 0.85));
  margin-bottom: 20px !important;
  font-size: 1rem !important;
  line-height: 1.75 !important;
  flex-grow: 1;
}

.article-card .ant-tag {
  font-size: 0.875rem;
  padding: 4px 12px;
  height: auto;
  line-height: 1.5;
  border-radius: 6px;
  font-weight: 500;
  border: none;
}

.article-card .ant-space {
  margin-bottom: 12px;
}

.article-metadata {
  color: var(--text-secondary, rgba(0, 0, 0, 0.65));
  font-size: 0.9rem;
  margin-top: auto;
  padding-top: 16px;
  border-top: 1px solid var(--border-color, #f0f0f0);
}

.article-card .ant-btn {
  height: 36px;
  padding: 0 16px;
  font-size: 0.95rem;
  border-radius: 8px;
  font-weight: 500;
}

.article-card .ant-btn-primary {
  background: var(--primary-color, #1890ff);
  border: none;
  box-shadow: 0 2px 4px rgba(24, 144, 255, 0.2);
}

.article-card .ant-btn-primary:hover {
  background: var(--primary-color-hover, #40a9ff);
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(24, 144, 255, 0.3);
}

/* Modal styles */
.article-modal-content {
  font-size: 1.1rem !important;
  line-height: 1.8 !important;
  color: rgba(0, 0, 0, 0.85) !important;
  padding: 0 16px !important;
}

.article-modal-content p {
  margin-bottom: 1.5em !important;
}

.article-modal-header {
  background: #fafafa !important;
  padding: 24px !important;
  border-radius: 12px !important;
  margin-bottom: 24px !important;
}

/* Fix for article snippet text color */
.article-snippet, 
.article-modal .ant-modal-body p,
.prose,
.ant-modal-body .prose p,
.ant-modal-body .article-modal-content {
  color: rgba(0, 0, 0, 0.85) !important;
  font-size: 1rem !important;
  line-height: 1.6 !important;
}

/* Additional text color fixes */
.ant-modal-body .prose * {
  color: rgba(0, 0, 0, 0.85) !important;
}

.ant-modal-body span, 
.ant-modal-body div,
.ant-modal-body li,
.ant-modal-body blockquote {
  color: rgba(0, 0, 0, 0.85) !important;
}

.ant-modal-body .text-gray-600,
.ant-modal-body .text-gray-500,
.ant-modal-body .text-gray-400 {
  color: rgba(0, 0, 0, 0.65) !important;
}

.ant-modal-body .text-xs {
  color: rgba(0, 0, 0, 0.65) !important;
}

/* Analysis Modal Styles */
.analysis-content {
  padding: 16px !important;
  background: #fff !important;
  border-radius: 8px !important;
  margin: 16px !important;
}

.analysis-content .ant-tag {
  margin-bottom: 12px !important;
  font-size: 0.85rem !important;
  padding: 2px 10px !important;
  border-radius: 4px !important;
}

/* Analysis Modal */
.analysis-modal .ant-modal-content {
  overflow: hidden !important;
  border-radius: 12px !important;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
}

.analysis-modal .ant-modal-header {
  background-color: #f9fafb !important;
  border-bottom: 1px solid #f0f0f0 !important;
  padding: 16px 24px !important;
}

.analysis-modal .ant-modal-body {
  padding: 0 !important;
  background-color: #f9fafb !important;
  max-height: 70vh !important;
  overflow-y: auto !important;
}

.analysis-modal .ant-modal-footer {
  border-top: 1px solid #f0f0f0 !important;
  padding: 12px 24px !important;
  text-align: right !important;
}

.analysis-modal-title {
  display: flex !important;
  align-items: center !important;
}

.analysis-modal-title .anticon {
  margin-right: 12px !important;
}

.analysis-modal-title .ant-typography {
  margin: 0 !important;
  color: #1f2937 !important;
  font-size: 1.25rem !important;
}

.analysis-metadata {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
  margin-bottom: 16px !important;
}

/* Markdown content */
.markdown-content {
  line-height: 1.8 !important;
  color: rgba(0, 0, 0, 0.85) !important;
  font-size: 1rem !important;
  padding: 16px 24px !important;
  background-color: white !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
  margin: 0 16px 16px 16px !important;
}

.markdown-content h1, 
.markdown-content h2, 
.markdown-content h3, 
.markdown-content h4 {
  margin-top: 1.5em !important;
  margin-bottom: 0.5em !important;
  font-weight: 600 !important;
  color: #1f2937 !important;
  line-height: 1.4 !important;
}

.markdown-content h1 {
  font-size: 1.8rem !important;
}

.markdown-content h2 {
  font-size: 1.5rem !important;
}

.markdown-content h3 {
  font-size: 1.3rem !important;
}

.markdown-content h4 {
  font-size: 1.1rem !important;
}

.markdown-content p {
  margin-bottom: 1.25em !important;
  line-height: 1.7 !important;
}

.markdown-content ul, 
.markdown-content ol {
  margin-bottom: 1.25em !important;
  padding-left: 1.5em !important;
  list-style-position: outside !important;
}

.markdown-content li {
  margin-bottom: 0.5em !important;
}

.markdown-content blockquote {
  border-left: 4px solid #e5e7eb !important;
  padding-left: 1em !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
  font-style: italic !important;
  color: #4b5563 !important;
}

.markdown-content pre {
  background-color: #f3f4f6 !important;
  padding: 1em !important;
  border-radius: 6px !important;
  overflow-x: auto !important;
  margin-bottom: 1.25em !important;
  font-family: monospace !important;
}

.markdown-content code {
  background-color: #f3f4f6 !important;
  padding: 0.2em 0.4em !important;
  border-radius: 3px !important;
  font-family: monospace !important;
  font-size: 0.9em !important;
}

.markdown-content table {
  width: 100% !important;
  border-collapse: collapse !important;
  margin-bottom: 1.25em !important;
}

.markdown-content th, 
.markdown-content td {
  border: 1px solid #e5e7eb !important;
  padding: 0.5em !important;
  text-align: left !important;
}

.markdown-content th {
  background-color: #f9fafb !important;
  font-weight: 600 !important;
}

/* Sources list and links */
.sources-list {
  list-style-type: disc !important;
  padding-left: 2em !important;
  margin-top: 0.5em !important;
  margin-bottom: 16px !important;
  margin-left: 16px !important;
  margin-right: 16px !important;
}

.sources-list li {
  margin-bottom: 0.75em !important;
  line-height: 1.6 !important;
}

.source-link {
  color: #1890ff !important;
  text-decoration: none !important;
  font-weight: 500 !important;
  transition: color 0.3s !important;
}

.source-link:hover {
  color: #40a9ff !important;
  text-decoration: underline !important;
}

.source-metadata {
  font-size: 0.9em !important;
  color: rgba(0, 0, 0, 0.45) !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .article-card .ant-card-body {
    padding: 16px;
  }

  .article-title {
    font-size: 1.1rem !important;
  }

  .article-description {
    font-size: 0.95rem !important;
  }

  .article-image-container {
    height: 180px;
  }
  
  .markdown-content {
    font-size: 0.95rem;
  }
  
  .markdown-content h1 {
    font-size: 1.5rem;
  }
  
  .markdown-content h2 {
    font-size: 1.3rem;
  }
  
  .markdown-content h3 {
    font-size: 1.1rem;
  }
  
  .markdown-content h4 {
    font-size: 1rem;
  }
}

@media (max-width: 480px) {
  .article-image-container {
    height: 160px;
  }
}

/* High contrast mode */
@media (prefers-contrast: high) {
  .article-title {
    color: #000000 !important;
  }

  .article-description {
    color: #000000 !important;
  }

  .article-card {
    border: 2px solid #000000;
  }
  
  .markdown-content {
    color: #000000;
  }
  
  .markdown-content blockquote {
    border-left-color: #000000;
    color: #000000;
  }
}

.analysis-content-wrapper {
  background-color: #f9fafb !important;
}

.analysis-loading-container {
  padding: 48px 24px !important;
  text-align: center !important;
  background-color: #fff !important;
  border-radius: 8px !important;
  margin: 16px !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

.analysis-error-container {
  padding: 24px !important;
  margin: 16px !important;
}

.analysis-sources-container {
  background-color: #fff !important;
  border-radius: 8px !important;
  margin: 0 16px 16px 16px !important;
  padding-top: 8px !important;
  padding-bottom: 16px !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

/* Make sure this gets added to the Responsive section */
@media (max-width: 768px) {
  .analysis-content {
    padding: 12px !important;
    margin: 12px !important;
  }
  
  .markdown-content {
    padding: 12px 16px !important;
  }
  
  .analysis-sources-container {
    margin: 0 12px 12px 12px !important;
  }
  
  .sources-list {
    padding-left: 1.5em !important;
  }
}
`; 