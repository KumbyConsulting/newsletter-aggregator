'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  Layout, 
  Typography, 
  Card, 
  Row, 
  Col, 
  Spin, 
  Alert, 
  Empty, 
  Space, 
  Tag, 
  Button, 
  Statistic, 
  Skeleton, 
  Tooltip, 
  Divider 
} from 'antd';
import {
  LinkOutlined,
  FileTextOutlined,
  GlobalOutlined,
  ReloadOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { getSources } from '@/services/api';

const { Content } = Layout;
const { Title, Text, Paragraph } = Typography;

interface SourceData {
  name: string;
  url: string;
  count: number;
  description?: string;
  logo_url?: string;
}

export default function SourcesPage() {
  const [sources, setSources] = useState<SourceData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSources = async () => {
    try {
      setLoading(true);
      const response = await getSources();
      
      if (response && response.sources) {
        setSources(response.sources || []);
        setError(null);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error fetching sources:', error);
      setError('Failed to load sources. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSources();
  }, []);

  // Source descriptions (in a real app, these would come from the API)
  const sourceDescriptions: Record<string, string> = {
    'Pharmaceutical Technology': 'Reporting on the pharmaceutical industry, including manufacturing technology, drug development, and market trends.',
    'FiercePharma': 'Daily updates on pharmaceutical industry news, market developments, and regulatory changes in the pharma sector.',
    'FDA News': 'Official news and announcements from the U.S. Food and Drug Administration regarding drug approvals and regulations.',
    'BioPharmaDive': 'In-depth analysis and news coverage of the biopharmaceutical industry and healthcare sector.',
    'DrugDiscoveryToday': 'Scientific journal covering research articles and news on drug discovery, development, and evaluation.',
    'PharmaTimes': 'News, insights, and analysis on the pharmaceutical and healthcare industries in the UK and globally.',
    'European Pharmaceutical Review': 'Coverage of European pharmaceutical regulations, developments, and industry news.',
    'PharmaManufacturing': 'Information on pharmaceutical manufacturing processes, technology, and regulatory compliance.',
    
    // Medical Journals
    'The Lancet': 'One of the world\'s oldest and most prestigious medical journals publishing original research and reviews.',
    'New England Journal of Medicine': 'Leading medical journal publishing research, reviews, and editorials on medical science and practice.',
    'JAMA': 'Peer-reviewed medical journal published by the American Medical Association covering all aspects of medical science.',
    'Nature Biotechnology': 'Research and news coverage on biotechnology, including pharmaceutical applications and innovations.',
    'British Medical Journal': 'Leading general medical journal covering clinical research, medical education, and global health policy.',
    'The Lancet Oncology': 'Specialized journal publishing high-quality research, reviews, and other articles in oncology.',
    'Science Translational Medicine': 'Publication focused on translational research to advance human health and medicine.',
    'Nature Medicine': 'Peer-reviewed journal publishing research on all aspects of medicine with an emphasis on understanding disease pathogenesis.',
    'Cell': 'Leading biomedical journal covering cellular and molecular biology, developmental biology, and beyond.',
    
    // Regulatory News
    'EMA News': 'Official news from the European Medicines Agency on drug approvals and regulatory updates in Europe.',
    'WHO News': 'Global health updates and pharmaceutical-related news from the World Health Organization.',
    'CDC': 'Public health information and updates from the U.S. Centers for Disease Control and Prevention.',
    
    // NewsAPI Integration
    'NewsAPI_Pharma': 'Aggregated pharmaceutical news from various sources via NewsAPI.',
    'NewsAPI_BioTech': 'Latest biotechnology news and developments from multiple sources via NewsAPI.',
    'NewsAPI_Clinical': 'Updates on clinical trials and research from various news outlets via NewsAPI.',
    'NewsAPI_Drug': 'Coverage of drug development news from multiple sources via NewsAPI.',
  };

  // Group sources by category
  const categorizeSource = (sourceName: string): string => {
    const lowerName = sourceName.toLowerCase();
    if (lowerName.includes('fda') || lowerName.includes('ema') || lowerName.includes('regulatory')) {
      return 'Regulatory';
    } else if (lowerName.includes('clinical') || lowerName.includes('trial')) {
      return 'Clinical Trials';
    } else if (lowerName.includes('journal') || lowerName.includes('lancet') || lowerName.includes('medicine') || lowerName.includes('cell')) {
      return 'Medical Journals';
    } else if (lowerName.includes('newsapi')) {
      return 'NewsAPI Integration';
    } else {
      return 'Industry News';
    }
  };

  // Group sources into categories
  const getGroupedSources = () => {
    const groups: Record<string, SourceData[]> = {};
    
    sources.forEach(source => {
      const category = categorizeSource(source.name);
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(source);
    });
    
    return groups;
  };

  // Render source cards
  const renderSourceCard = (source: SourceData) => {
    const description = source.description || sourceDescriptions[source.name] || `News source for various topics.`;
    
    return (
      <Card 
        hoverable 
        className="source-card"
        actions={[
          <Link href={`/?source=${encodeURIComponent(source.name)}`} key="view">
            <Space>
              <FileTextOutlined />
              View articles
            </Space>
          </Link>,
          <a 
            href={source.url} 
            target="_blank" 
            rel="noopener noreferrer"
            key="website"
          >
            <Space>
              <LinkOutlined />
              Visit website
            </Space>
          </a>
        ]}
      >
        <Card.Meta
          avatar={
            source.logo_url ? (
              <img 
                src={source.logo_url} 
                alt={`${source.name} logo`} 
                className="source-logo"
                style={{ width: 40, height: 40, objectFit: 'contain', borderRadius: '50%', background: '#f5f5f5', padding: 4 }}
              />
            ) : (
              <div style={{ width: 40, height: 40, borderRadius: '50%', background: '#1890ff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <GlobalOutlined style={{ color: 'white', fontSize: 20 }} />
              </div>
            )
          }
          title={source.name}
          description={
            <Space direction="vertical" style={{ width: '100%' }}>
              <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 8 }}>
                {description}
              </Paragraph>
              <Statistic 
                value={source.count} 
                suffix="articles" 
                valueStyle={{ fontSize: 16 }} 
              />
            </Space>
          }
        />
      </Card>
    );
  };

  // Group sources into categories for display
  const groupedSources = getGroupedSources();

  return (
    <Layout className="layout">
      <Content className="site-layout-content" style={{ padding: '0 50px' }}>
        <div className="container" style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 24, marginBottom: 16 }}>
            <Title level={2}>
              News Sources
            </Title>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={fetchSources}
              loading={loading}
              type="primary"
            >
              Refresh
            </Button>
          </div>
          
          <Paragraph style={{ marginBottom: 24 }}>
            Browse all available news sources grouped by category. Click on a source to view its articles or visit its website.
          </Paragraph>

          {error && (
            <Alert
              message="Error Loading Sources"
              description={error}
              type="error"
              showIcon
              style={{ marginBottom: 24 }}
            />
          )}
          
          {loading ? (
            <Row gutter={[24, 24]}>
              {[...Array(9)].map((_, i) => (
                <Col key={i} xs={24} md={12} lg={8}>
                  <Card>
                    <Skeleton active avatar paragraph={{ rows: 2 }} />
                    <div style={{ marginTop: 16 }}>
                      <Skeleton.Button active size="small" style={{ width: 100, marginRight: 16 }} />
                      <Skeleton.Button active size="small" style={{ width: 100 }} />
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          ) : sources.length === 0 ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <span>
                  No sources found. Please check the connection to the backend service.
                </span>
              }
            >
              <Button type="primary" onClick={fetchSources}>Try Again</Button>
            </Empty>
          ) : (
            <>
              {Object.entries(groupedSources).map(([category, categorySources]) => (
                <div key={category} style={{ marginBottom: 32 }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                    <Title level={3} style={{ margin: 0 }}>{category}</Title>
                    <Tag color="blue" style={{ marginLeft: 8 }}>{categorySources.length}</Tag>
                  </div>
                  
                  <Row gutter={[24, 24]}>
                    {categorySources.map((source) => (
                      <Col key={source.name} xs={24} md={12} lg={8}>
                        {renderSourceCard(source)}
                      </Col>
                    ))}
                  </Row>
                  
                  <Divider style={{ marginTop: 32, marginBottom: 32 }} />
                </div>
              ))}
            </>
          )}
        </div>
      </Content>
      
      <Layout.Footer style={{ textAlign: 'center', background: '#f0f2f5', marginTop: 24 }}>
        <Text type="secondary">© {new Date().getFullYear()} Newsletter Aggregator. All rights reserved.</Text>
      </Layout.Footer>
    </Layout>
  );
} 