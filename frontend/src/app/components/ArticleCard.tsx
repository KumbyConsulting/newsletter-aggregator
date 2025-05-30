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
import { formatDate, getReadingTime } from '../../../../utils/article';

const { Title, Text, Paragraph } = Typography;

interface ArticleCardProps {
  article: Article;
  className?: string;
  highlight?: string;
  highlighted?: boolean;
}

// Highlight utility
function highlightText(text: string, query: string): React.ReactNode {
  if (!query) return text;
  const safeQuery = query.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&');
  const regex = new RegExp(`(${safeQuery})`, 'gi');
  const parts = text.split(regex);
  return parts.map((part, i) =>
    regex.test(part) ? <mark key={i}>{part}</mark> : part
  );
}

export const ArticleCard: React.FC<ArticleCardProps> = ({ article, className, highlight, highlighted }) => {
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

  // Fallback image logic
  const renderImage = () => {
    if (image_url) {
      return (
        <div className="article-card__image" onClick={showModal}>
          <Image
            src={image_url}
            alt={title}
            layout="fill"
            objectFit="cover"
            className="article-image"
            onError={(e: any) => { e.target.onerror = null; e.target.src = '/fallback-image.svg'; }}
          />
        </div>
      );
    }
    return (
      <div className="article-card__image-placeholder" onClick={showModal}>
        <FileTextOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />
      </div>
    );
  };

  return (
    <Card
      hoverable
      className={`article-card${highlighted ? ' article-card--highlighted' : ''}${className ? ' ' + className : ''}`}
      cover={renderImage()}
      bodyStyle={{ padding: 0, display: 'flex', flexDirection: 'column', height: '100%' }}
    >
      <div className="article-card__content">
        {/* Title */}
        <Title
          level={4}
          className="article-card__title"
          ellipsis={{ rows: 2 }}
          title={metadata.title}
        >
          <a href={metadata.link} target="_blank" rel="noopener noreferrer">
            {highlightText(metadata.title, highlight || '')}
          </a>
        </Title>

        {/* Description */}
        <Paragraph ellipsis={{ rows: 2 }} className="article-card__description">
          {highlightText(snippetForCard || '', highlight || '')}
        </Paragraph>

        {/* Metadata Footer */}
        <div className="article-card__footer">
          <span className="article-meta-source"><BookOutlined /> {metadata.source}</span>
          <span className="article-meta-dot">•</span>
          <span className="article-meta-date"><CalendarOutlined /> {formatDate(metadata.pub_date)}</span>
          {metadata.reading_time && <><span className="article-meta-dot">•</span><span className="article-meta-reading"><ClockCircleOutlined /> {getReadingTime(metadata.description, metadata.reading_time)} min</span></>}
        </div>

        {/* Actions Footer */}
        <div className="article-card__footer" style={{ justifyContent: 'flex-end', gap: 8, marginTop: 10 }}>
          <Tooltip title="View Details">
            <Button
              type="primary"
              size="small"
              icon={<EyeOutlined />}
              onClick={showModal}
              aria-label="View Details"
            />
          </Tooltip>
          <Tooltip title="Open original article">
            <Button
              type="text"
              size="small"
              href={metadata.link}
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
          <Tooltip title="Analyze">
            <Button
              type="text"
              size="small"
              icon={<BulbOutlined />}
              onClick={handleAnalyzeClick}
              loading={loading}
              aria-label="Analyze"
            />
          </Tooltip>
        </div>
      </div>

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
          <Button key="link" type="primary" href={metadata.link} target="_blank" rel="noopener noreferrer" icon={<LinkOutlined />}>
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
              <Space><CalendarOutlined /> <Text>{formatDate(pub_date)}</Text></Space>
              {metadata.relevance_score !== undefined && (
                <>• <Space><SearchOutlined /> <Text>Relevance: {(metadata.relevance_score * 100).toFixed(0)}%</Text></Space></>
              )}
            </Space>
            
            {/* Source Link */}
            <Text className="text-xs">
              <LinkOutlined className="mr-1" />
              <a href={metadata.link} target="_blank" rel="noopener noreferrer">{metadata.link}</a>
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
    </Card>
  );
};

// Loading skeleton for the article card
export const ArticleCardSkeleton: React.FC = () => {
  return (
    <Card loading className="article-card-skeleton" />
  );
}; 