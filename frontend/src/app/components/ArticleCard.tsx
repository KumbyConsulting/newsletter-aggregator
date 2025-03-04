'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Card, Tag, Typography, Button, Modal, message, Space, Divider, Tooltip, Skeleton } from 'antd';
import { 
  ReadOutlined, 
  ShareAltOutlined, 
  ClockCircleOutlined, 
  CloseOutlined, 
  ExportOutlined, 
  AppstoreOutlined, 
  FileTextOutlined
} from '@ant-design/icons';

const { Title, Paragraph, Text } = Typography;

// Article Card Props
export interface ArticleProps {
  id: string;
  title?: string;
  description?: string;
  link?: string;
  pub_date?: string;
  source?: string;
  topic?: string;
  summary?: string;
  image_url?: string;
  metadata?: any; // Add metadata for nested structure
}

// Static skeleton loader for ArticleCard
export const ArticleCardSkeleton = () => (
  <Card
    hoverable
    style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
    styles={{ body: { flex: 1, display: 'flex', flexDirection: 'column' } }}
    cover={
      <Skeleton.Image 
        style={{ 
          height: 176,
          width: '100%'
        }} 
        active 
      />
    }
  >
    <Skeleton active paragraph={{ rows: 3 }} />
    <Divider style={{ margin: '12px 0' }} />
    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 'auto' }}>
      <Skeleton.Button active style={{ width: 120 }} />
      <Space>
        <Skeleton.Button active shape="circle" style={{ width: 32 }} />
        <Skeleton.Button active shape="circle" style={{ width: 32 }} />
      </Space>
    </div>
  </Card>
);

export default function ArticleCard({
  id,
  title,
  description,
  link,
  pub_date,
  source,
  topic,
  summary,
  image_url,
  metadata
}: ArticleProps) {
  const [showFullContent, setShowFullContent] = useState(false);
  const [similarArticles, setSimilarArticles] = useState<any[]>([]);
  const [showSimilarModal, setShowSimilarModal] = useState(false);
  
  // Handle both flat and nested data structures
  const meta = metadata || {};
  const finalTitle = title || meta.title || 'Untitled';
  const finalDescription = description || meta.description || '';
  const finalLink = link || meta.link || '#';
  const finalPubDate = pub_date || meta.pub_date || 'Unknown date';
  const finalSource = source || meta.source || 'Unknown source';
  const finalTopic = topic || meta.topic || 'Uncategorized';
  const finalSummary = summary || meta.summary;
  const finalImageUrl = image_url || meta.image_url;
  
  // Clean description if it contains HTML
  const cleanDescription = finalDescription ? finalDescription.replace(/<[^>]*>?/gm, '') : 'No description available';
  
  // Function to toggle full content visibility
  const toggleContent = () => {
    setShowFullContent(!showFullContent);
  };
  
  // Function to share article
  const shareArticle = () => {
    if (navigator.share) {
      navigator.share({
        title: finalTitle,
        url: finalLink
      }).catch(console.error);
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(finalLink).then(() => {
        message.success('Link copied to clipboard!');
      }).catch(console.error);
    }
  };
  
  // Function to show similar articles
  const fetchSimilarArticles = async () => {
    try {
      const response = await fetch(`/api/similar-articles/${id}`);
      const data = await response.json();
      setSimilarArticles(data.articles || []);
      setShowSimilarModal(true);
    } catch (error) {
      console.error('Error:', error);
      message.error('Error fetching similar articles');
    }
  };
  
  // Close similar articles modal
  const closeSimilarModal = () => {
    setShowSimilarModal(false);
  };

  // Generate cover for Card
  const cardCover = finalImageUrl ? (
    <div 
      style={{ 
        height: '176px', 
        backgroundImage: `url(${finalImageUrl})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        position: 'relative'
      }}
    >
      <div style={{ position: 'absolute', top: '12px', left: '12px' }}>
        <Tag color="blue">{finalTopic}</Tag>
      </div>
    </div>
  ) : (
    <div 
      style={{ 
        height: '176px', 
        background: 'linear-gradient(to right, #1890ff, #096dd9)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        position: 'relative'
      }}
    >
      <FileTextOutlined style={{ fontSize: '48px', color: 'white' }} />
      <div style={{ position: 'absolute', top: '12px', left: '12px' }}>
        <Tag color="blue">{finalTopic}</Tag>
      </div>
    </div>
  );

  return (
    <>
      <Card
        hoverable
        cover={cardCover}
        style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
        styles={{ body: { flex: 1, display: 'flex', flexDirection: 'column' } }}
      >
        {/* Source and Date */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
          <Text type="secondary">
            <ReadOutlined style={{ marginRight: 4 }} />
            {finalSource}
          </Text>
          <Text type="secondary">
            <ClockCircleOutlined style={{ marginRight: 4 }} />
            {finalPubDate}
          </Text>
        </div>
        
        {/* Title */}
        <Title level={4} ellipsis={{ rows: 2 }} style={{ marginBottom: 12 }}>
          <a href={finalLink} target="_blank" rel="noopener noreferrer">
            {finalTitle}
          </a>
        </Title>
        
        {/* Description */}
        <div style={{ marginBottom: 16, flex: 1 }}>
          {finalDescription && finalDescription.length > 200 ? (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <Text strong type="secondary" style={{ textTransform: 'uppercase', fontSize: '12px' }}>
                  Full Article
                </Text>
                <Button 
                  type="link" 
                  size="small" 
                  onClick={toggleContent}
                  style={{ padding: 0 }}
                >
                  {showFullContent ? 'Hide Full Content' : 'Show Full Content'}
                </Button>
              </div>
              {showFullContent ? (
                <div style={{ maxHeight: '240px', overflowY: 'auto' }}>
                  <div dangerouslySetInnerHTML={{ __html: finalDescription }} />
                </div>
              ) : (
                <Paragraph ellipsis={{ rows: 3 }} style={{ fontSize: '14px' }}>
                  {cleanDescription}
                </Paragraph>
              )}
            </>
          ) : (
            <Paragraph style={{ fontSize: '14px' }}>
              {cleanDescription}
            </Paragraph>
          )}
        </div>
        
        {/* Summary (if available) */}
        {finalSummary && (
          <div 
            style={{ 
              background: '#f0f5ff', 
              padding: 16, 
              borderRadius: 8, 
              marginBottom: 16,
              borderLeft: '4px solid #1890ff'
            }}
          >
            <Text strong type="secondary" style={{ textTransform: 'uppercase', fontSize: '12px', display: 'block', marginBottom: 8 }}>
              Summary
            </Text>
            <Paragraph style={{ fontSize: '14px', margin: 0 }}>
              {finalSummary}
            </Paragraph>
          </div>
        )}
        
        {/* Actions Footer */}
        <Divider style={{ margin: '12px 0' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'auto' }}>
          <Button 
            type="primary" 
            icon={<ExportOutlined />}
            href={finalLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            Read Article
          </Button>
          
          <Space>
            <Tooltip title="Find similar articles">
              <Button 
                type="text" 
                icon={<AppstoreOutlined />} 
                onClick={fetchSimilarArticles}
                shape="circle"
              />
            </Tooltip>
            
            <Tooltip title="Share article">
              <Button 
                type="text" 
                icon={<ShareAltOutlined />} 
                onClick={shareArticle}
                shape="circle"
              />
            </Tooltip>
          </Space>
        </div>
      </Card>
      
      {/* Similar Articles Modal */}
      <Modal
        title="Similar Articles"
        open={showSimilarModal}
        onCancel={closeSimilarModal}
        footer={null}
        width={800}
        closeIcon={<CloseOutlined />}
      >
        <div style={{ maxHeight: '60vh', overflowY: 'auto' }}>
          {!similarArticles || similarArticles.length === 0 ? (
            <Text type="secondary">No similar articles found.</Text>
          ) : (
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              {similarArticles.map((article, index) => {
                const meta = article.metadata || {};
                return (
                  <Card key={index} size="small" style={{ width: '100%' }}>
                    <Title level={5} style={{ marginTop: 0 }}>
                      {meta.title || article.title || 'Untitled'}
                    </Title>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Text type="secondary">{meta.source || article.source || 'Unknown source'}</Text>
                      <Text type="secondary">{meta.pub_date || article.pub_date || 'Unknown date'}</Text>
                    </div>
                    
                    <div style={{ marginBottom: 12 }}>
                      <Tag color="blue">{meta.topic || article.topic || 'Uncategorized'}</Tag>
                    </div>
                    
                    <Button 
                      type="link" 
                      icon={<ExportOutlined />} 
                      href={meta.link || article.link || '#'}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ padding: 0 }}
                    >
                      Read Article
                    </Button>
                  </Card>
                );
              })}
            </Space>
          )}
        </div>
      </Modal>
    </>
  );
} 