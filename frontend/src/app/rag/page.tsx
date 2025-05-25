'use client';

import React, { useState, useEffect, useRef, Dispatch, SetStateAction } from 'react';
import {
  Layout,
  Typography,
  Card,
  Input,
  Button,
  List,
  Spin,
  Divider,
  Space,
  Tag,
  Alert,
  Tabs,
  Empty,
  Switch,
  Tooltip,
  message,
  Select,
  Badge,
  Row,
  Col,
  Menu,
  Tree,
  Pagination
} from 'antd';
import {
  SendOutlined,
  ClockCircleOutlined,
  ClearOutlined,
  InfoCircleOutlined,
  HistoryOutlined,
  DeleteOutlined,
  SearchOutlined,
  ExperimentOutlined,
  MedicineBoxOutlined,
  FileProtectOutlined,
  SafetyCertificateOutlined,
  FundOutlined,
  ApiOutlined,
  TeamOutlined,
  RocketOutlined,
  AimOutlined,
  BranchesOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  QuestionCircleOutlined,
  SolutionOutlined,
  ThunderboltOutlined,
  BulbOutlined,
  NodeIndexOutlined,
  SaveOutlined,
  ReloadOutlined,
  FolderOpenOutlined
} from '@ant-design/icons';
import {
  submitRagQuery,
  streamRagQuery,
  getRagHistory,
  clearRagHistory,
  saveAnalysis,
  getSavedAnalyses,
  deleteSavedAnalysis,
  RAGQuery,
  RAGResponse,
  RAGHistoryItem,
  SavedAnalysis,
  Article,
  AnalysisType
} from '@/services/api';
import ReactMarkdown from 'react-markdown';
import { formatDistanceToNow } from 'date-fns';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Header, Content } = Layout;
const { Option } = Select;

// Update type definitions
type ViewType = 'history' | 'analysis' | 'saved';
type ExtendedAnalysisType = AnalysisType | ViewType;

// Define extended types that build on the API types
interface ExtendedRAGQuery extends Omit<RAGQuery, 'analysis_type'> {
  analysis_type: ExtendedAnalysisType;
}

interface ExtendedRAGHistoryItem extends Omit<RAGHistoryItem, 'analysis_type'> {
  analysis_type: ExtendedAnalysisType;
}

// Analysis type options with improved categorization
const ANALYSIS_TYPES: { [key in ExtendedAnalysisType]: { label: string; icon: any; description: string; category: string } } = {
  // View types
  history: {
    label: 'History',
    icon: <HistoryOutlined />,
    description: 'View previous analyses and queries',
    category: 'Navigation'
  },
  analysis: {
    label: 'Analysis',
    icon: <SearchOutlined />,
    description: 'Perform new analysis',
    category: 'Navigation'
  },
  saved: {
    label: 'Saved Analyses',
    icon: <SaveOutlined />,
    description: 'View and manage saved analyses',
    category: 'Navigation'
  },
  // Analysis types
  regulatory: {
    label: 'Regulatory Analysis',
    icon: <FileProtectOutlined />,
    description: 'Analyze regulatory impacts, compliance requirements, and policy changes',
    category: 'Compliance & Safety'
  },
  safety: {
    label: 'Safety Analysis',
    icon: <SafetyCertificateOutlined />,
    description: 'Evaluate drug safety profiles, adverse events, and risk management',
    category: 'Compliance & Safety'
  },
  clinical: {
    label: 'Clinical Analysis',
    icon: <ExperimentOutlined />,
    description: 'Analyze clinical trial data, outcomes, and research findings',
    category: 'Research & Development'
  },
  pipeline: {
    label: 'Pipeline Analysis',
    icon: <AimOutlined />,
    description: 'Track drug development pipelines and therapeutic programs',
    category: 'Research & Development'
  },
  market: {
    label: 'Market Analysis',
    icon: <FundOutlined />,
    description: 'Analyze market trends, competition, and commercial opportunities',
    category: 'Business & Market'
  },
  competitive: {
    label: 'Competitive Intelligence',
    icon: <TeamOutlined />,
    description: 'Monitor competitor activities and strategic developments',
    category: 'Business & Market'
  },
  manufacturing: {
    label: 'Manufacturing',
    icon: <ApiOutlined />,
    description: 'Analyze manufacturing processes, quality control, and supply chain',
    category: 'Operations'
  },
  digital: {
    label: 'Digital Health',
    icon: <RocketOutlined />,
    description: 'Explore digital therapeutics, health tech, and innovation',
    category: 'Innovation'
  }
};

// Group analysis types by category
const CATEGORIES = Object.entries(ANALYSIS_TYPES).reduce((acc, [key, value]) => {
  if (!acc[value.category]) {
    acc[value.category] = [];
  }
  acc[value.category].push({ key, ...value });
  return acc;
}, {} as Record<string, Array<any>>);

// Add new interfaces for structured content
interface AnalysisNode {
  key: string;
  title: string;
  children?: AnalysisNode[];
  icon?: React.ReactNode;
  content?: string;
}

// Add interface for tree node return type
interface TreeNodeData {
  key: string;
  title: React.ReactNode;
  children?: TreeNodeData[];
}

interface ViewProps {
  activeView: ViewType;
  setActiveView: Dispatch<SetStateAction<ViewType>>;
  history: ExtendedRAGHistoryItem[];
  handleClearHistory: () => void;
  loadHistoryItem: (item: ExtendedRAGHistoryItem) => void;
}

const renderConfidenceIndicator = (confidence: number) => (
  <Tag color={confidence > 0.7 ? 'green' : confidence > 0.4 ? 'orange' : 'red'}>
    {Math.round(confidence * 100)}% confidence
  </Tag>
);

const HistoryView: React.FC<ViewProps> = ({
  activeView,
  setActiveView,
  history,
  handleClearHistory,
  loadHistoryItem,
}) => (
  <Card>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
      <Title level={4} style={{ margin: 0 }}>Previous Analyses</Title>
      <Button 
        onClick={handleClearHistory}
        icon={<DeleteOutlined />} 
        disabled={!history.length}
        type="primary"
        danger
      >
        Clear History
      </Button>
    </div>

    {history.length === 0 ? (
      <Empty 
        description={
          <span>
            No analysis history yet. <br />
            Try running an analysis first.
          </span>
        }
        image={Empty.PRESENTED_IMAGE_SIMPLE}
      />
    ) : (
      <List
        itemLayout="vertical"
        dataSource={history}
        renderItem={(item) => (
          <List.Item
            className="history-item"
            style={{ 
              marginBottom: 16, 
              padding: 16, 
              borderRadius: 8,
              border: '1px solid #f0f0f0',
              transition: 'all 0.3s ease',
              cursor: 'pointer',
              background: '#fff'
            }}
            onClick={() => {
              loadHistoryItem(item);
              setActiveView('analysis');
            }}
            actions={[
              <Button 
                type="link" 
                onClick={(e) => {
                  e.stopPropagation();
                  loadHistoryItem(item);
                  setActiveView('analysis');
                }}
                icon={<SearchOutlined />}
              >
                Reuse Query
              </Button>
            ]}
          >
            <List.Item.Meta
              avatar={
                <div style={{ 
                  width: 40, 
                  height: 40, 
                  borderRadius: '50%', 
                  background: '#1890ff', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  color: 'white'
                }}>
                  {ANALYSIS_TYPES[item.analysis_type || 'regulatory'].icon}
                </div>
              }
              title={<Text strong>{item.query}</Text>}
              description={
                <Space direction="vertical" size={0}>
                  <Space>
                    <Tag color="blue">{ANALYSIS_TYPES[item.analysis_type || 'regulatory'].label}</Tag>
                    <span>
                      <ClockCircleOutlined style={{ marginRight: 4 }} />
                      <Text type="secondary">
                        {(() => {
                          try {
                            const dateObj = new Date(item.timestamp);
                            if (isNaN(dateObj.getTime())) {
                              return 'Invalid date';
                            }
                            return formatDistanceToNow(dateObj, { addSuffix: true });
                          } catch (e) {
                            console.error("Error formatting date:", item.timestamp, e);
                            return 'Invalid date';
                          }
                        })()}
                      </Text>
                    </span>
                    {item.confidence && renderConfidenceIndicator(item.confidence)}
                  </Space>
                  <div style={{ marginTop: 8 }}>
                    <Text type="secondary" ellipsis={true}>
                      {item.response.substring(0, 150)}...
                    </Text>
                  </div>
                </Space>
              }
            />
          </List.Item>
        )}
      />
    )}
  </Card>
);

// Add SavedAnalysisView component
interface SavedAnalysisViewProps {
  activeView: ViewType;
  setActiveView: Dispatch<SetStateAction<ViewType>>;
  loadSavedAnalysis: (analysis: SavedAnalysis) => void;
}

const SavedAnalysisView: React.FC<SavedAnalysisViewProps> = ({
  activeView,
  setActiveView,
  loadSavedAnalysis,
}) => {
  const [analyses, setAnalyses] = useState<SavedAnalysis[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (activeView === 'saved') {
      loadSavedAnalyses();
    }
  }, [activeView, page, pageSize]);

  const loadSavedAnalyses = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await getSavedAnalyses(
        pageSize, 
        (page - 1) * pageSize,
        'timestamp',
        'desc'
      );
      
      setAnalyses(response.analyses);
      setTotal(response.total);
    } catch (error) {
      console.error('Error loading saved analyses:', error);
      setError('Failed to load saved analyses');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAnalysis = async (analysisId: string) => {
    try {
      const result = await deleteSavedAnalysis(analysisId);
      
      if (result.success) {
        message.success('Analysis deleted successfully');
        loadSavedAnalyses(); // Reload the list
      } else {
        throw new Error('Failed to delete analysis');
      }
    } catch (error) {
      console.error('Error deleting analysis:', error);
      message.error('Failed to delete analysis');
    }
  };

  const handleLoadAnalysis = (analysis: SavedAnalysis) => {
    loadSavedAnalysis(analysis);
    setActiveView('analysis');
  };

  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={4} style={{ margin: 0 }}>Saved Analyses</Title>
        <Button 
          onClick={loadSavedAnalyses}
          icon={<ReloadOutlined />}
          disabled={loading}
          type="primary"
        >
          Refresh
        </Button>
      </div>

      {error && (
        <Alert 
          message="Error" 
          description={error} 
          type="error" 
          showIcon 
          style={{ marginBottom: 16 }} 
        />
      )}

      {loading ? (
        <div style={{ padding: 32, textAlign: 'center' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text type="secondary">Loading saved analyses...</Text>
          </div>
        </div>
      ) : analyses.length === 0 ? (
        <Empty 
          description={
            <span>
              No saved analyses found. <br />
              Save your analyses for future reference.
            </span>
          }
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <>
          <List
            itemLayout="vertical"
            dataSource={analyses}
            renderItem={(analysis) => (
              <List.Item
                className="saved-item"
                style={{ 
                  marginBottom: 16, 
                  padding: 16, 
                  borderRadius: 8,
                  border: '1px solid #f0f0f0',
                  transition: 'all 0.3s ease',
                  cursor: 'pointer',
                  background: '#fff'
                }}
                onClick={() => handleLoadAnalysis(analysis)}
                actions={[
                  <Button 
                    type="link" 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleLoadAnalysis(analysis);
                    }}
                    icon={<FolderOpenOutlined />}
                  >
                    Open
                  </Button>,
                  <Button 
                    type="link" 
                    danger
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteAnalysis(analysis.id);
                    }}
                    icon={<DeleteOutlined />}
                  >
                    Delete
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={
                    <div style={{ 
                      width: 40, 
                      height: 40, 
                      borderRadius: '50%', 
                      background: '#1890ff', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      color: 'white'
                    }}>
                      {ANALYSIS_TYPES[analysis.analysis_type as AnalysisType || 'regulatory'].icon}
                    </div>
                  }
                  title={<Text strong>{analysis.query}</Text>}
                  description={
                    <Space direction="vertical" size={0}>
                      <Space>
                        <Tag color="blue">
                          {ANALYSIS_TYPES[analysis.analysis_type as AnalysisType || 'regulatory'].label}
                        </Tag>
                        <span>
                          <ClockCircleOutlined style={{ marginRight: 4 }} />
                          <Text type="secondary">
                            {(() => {
                              try {
                                const dateObj = new Date(analysis.timestamp);
                                if (isNaN(dateObj.getTime())) {
                                  return 'Invalid date';
                                }
                                return formatDistanceToNow(dateObj, { addSuffix: true });
                              } catch (e) {
                                console.error("Error formatting date:", analysis.timestamp, e);
                                return 'Invalid date';
                              }
                            })()}
                          </Text>
                        </span>
                        {analysis.confidence && renderConfidenceIndicator(analysis.confidence)}
                      </Space>
                      <div style={{ marginTop: 8 }}>
                        <Text type="secondary" ellipsis={true}>
                          {analysis.response.substring(0, 150)}...
                        </Text>
                      </div>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
          
          {total > pageSize && (
            <div style={{ textAlign: 'right', marginTop: 24 }}>
              <Pagination
                current={page}
                pageSize={pageSize}
                total={total}
                onChange={setPage}
                onShowSizeChange={(current, size) => {
                  setPage(1);
                  setPageSize(size);
                }}
                showSizeChanger
                showQuickJumper
              />
            </div>
          )}
        </>
      )}
    </Card>
  );
};

// Add this adapter function after other imports but before component definitions
// This adapter helps with type compatibility between RAGSource and Article types
const adaptSourceToArticle = (source: any): Article => {
  return {
    id: source.id || `source-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    title: source.title || '',
    description: source.description || '',
    link: source.link || '',
    pub_date: source.pub_date || source.date || new Date().toISOString(),
    source: source.source || 'Unknown',
    topic: source.topic || 'Uncategorized',
    date: source.date || source.pub_date,
    summary: source.summary,
    // Add any other required fields with fallbacks
  };
};

const RAGPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [timeAware, setTimeAware] = useState(true);
  const [response, setResponse] = useState<string>('');
  const [sources, setSources] = useState<Article[]>([]);
  const [history, setHistory] = useState<ExtendedRAGHistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [confidence, setConfidence] = useState<number>(0);
  const [activeView, setActiveView] = useState<ViewType>('analysis');
  const [selectedAnalysisType, setSelectedAnalysisType] = useState<AnalysisType>('regulatory');
  const responseEndRef = useRef<HTMLDivElement>(null);
  const [savingAnalysis, setSavingAnalysis] = useState(false);

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    if (responseEndRef.current) {
      responseEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [response]);

  const loadHistory = async () => {
    try {
      const response = await getRagHistory();
      // Ensure history is an array
      const historyArray = Array.isArray(response) ? response : [];
      const extendedHistory: ExtendedRAGHistoryItem[] = historyArray.map(item => ({
        ...item,
        analysis_type: (item.analysis_type || 'regulatory') as AnalysisType
      }));
      setHistory(extendedHistory);
    } catch (error) {
      console.error('Failed to load history:', error);
      setError('Failed to load history. Please try again.');
    }
  };

  const handleAnalysisTypeChange = (value: string) => {
    setSelectedAnalysisType(value as AnalysisType);
  };

  const handleStreamQuery = async () => {
    if (!query.trim()) return;
    
    setError(null);
    setLoading(true);
    setResponse('');
    setSources([]);
    setConfidence(0);
    
    try {
      const ragQuery: ExtendedRAGQuery = {
        query: query.trim(),
        time_aware: timeAware,
        analysis_type: selectedAnalysisType
      };
      
      let accumulatedText = '';
      let buffer = '';
      const chunkBreakers = ['\n\n', '. ', ':**', '* '];
      
      await streamRagQuery(ragQuery as RAGQuery, (chunk) => {
        try {
          // Process each line as a separate JSON object
          const lines = chunk.split('\n').filter(line => line.trim());
          
          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              
              // Process based on the type field
              if (data.type === 'content') {
                // Handle content chunks
                if (data.content) {
                  buffer += data.content;
                  
                  // Check for natural break points
                  let shouldUpdate = false;
                  for (const breaker of chunkBreakers) {
                    if (buffer.includes(breaker)) {
                      const parts = buffer.split(breaker);
                      buffer = parts.pop() || '';
                      accumulatedText += parts.join(breaker) + (parts.length > 0 ? breaker : '');
                      shouldUpdate = true;
                    }
                  }
                  
                  // If buffer is getting too long, flush it
                  if (buffer.length > 100) {
                    accumulatedText += buffer;
                    buffer = '';
                    shouldUpdate = true;
                  }
                  
                  if (shouldUpdate) {
                    setResponse(accumulatedText);
                  }
                  
                  // If this is the final chunk, append any remaining buffer
                  if (data.done && buffer) {
                    accumulatedText += buffer;
                    setResponse(accumulatedText);
                  }
                }
              } else if (data.type === 'sources') {
                // Handle sources data with adapter
                const adaptedSources = Array.isArray(data.sources) 
                  ? data.sources.map(adaptSourceToArticle) 
                  : [];
                setSources(adaptedSources);
              } else if (data.type === 'metadata') {
                // Handle metadata including confidence score
                setConfidence(data.confidence || 0);
              } else if (data.type === 'error') {
                // Handle error messages
                throw new Error(data.error || 'Unknown error in stream');
              }
            } catch (lineError) {
              // If we can't parse a single line, try to continue with others
              console.warn('Error parsing JSON line:', lineError);
            }
          }
        } catch (error) {
          // This catches errors in the outer processing, not JSON parsing
          console.error('Error processing chunk:', error);
          
          // Handle raw text chunks if it's not valid JSON
          buffer += chunk;
          if (buffer.length > 100) {
            accumulatedText += buffer;
            buffer = '';
            setResponse(accumulatedText);
          }
        }
      });
      
      // Flush any remaining buffer at the end
      if (buffer.length > 0) {
        accumulatedText += buffer;
        setResponse(accumulatedText);
      }
      
      loadHistory();
    } catch (error) {
      console.error('Error streaming analysis:', error);
      setError('Failed to stream analysis. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClearHistory = async () => {
    try {
      await clearRagHistory();
      setHistory([]);
      message.success('Analysis history cleared successfully');
    } catch (error) {
      console.error('Error clearing history:', error);
      message.error('Failed to clear analysis history');
    }
  };

  const loadHistoryItem = (item: ExtendedRAGHistoryItem) => {
    setQuery(item.query);
    setResponse(item.response);
    
    // Ensure source type compatibility
    const adaptedSources = Array.isArray(item.sources)
      ? item.sources.map(s => adaptSourceToArticle(s))
      : [];
    setSources(adaptedSources);
    
    setConfidence(item.confidence || 0);
    setSelectedAnalysisType((item.analysis_type || 'regulatory') as AnalysisType);
    setActiveTab('chat');
  };

  const loadSavedAnalysis = (analysis: SavedAnalysis) => {
    setQuery(analysis.query);
    setResponse(analysis.response);
    
    // Ensure source type compatibility
    const adaptedSources = Array.isArray(analysis.sources)
      ? analysis.sources.map(s => adaptSourceToArticle(s))
      : [];
    setSources(adaptedSources);
    
    setConfidence(analysis.confidence || 0);
    setSelectedAnalysisType((analysis.analysis_type as AnalysisType) || 'regulatory');
    setActiveView('analysis');
  };

  const renderSources = (sources: Article[]) => {
    if (!sources.length) return null;

    return (
      <div style={{ marginTop: '16px' }}>
        <Divider orientation="left">
          <Space>
            <InfoCircleOutlined />
            <Text strong>Sources</Text>
          </Space>
        </Divider>
        <List
          size="small"
          dataSource={sources}
          renderItem={(source, index) => (
            <List.Item>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Space>
                  <Badge status="processing" text={`Source ${index + 1}`} />
                  <Tag color="blue">{source.source}</Tag>
                  {source.date && (
                    <Tag color="green">
                      {(() => {
                        try {
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
                    </Tag>
                  )}
                </Space>
                <a href={source.link} target="_blank" rel="noopener noreferrer">
                  {source.title}
                </a>
              </Space>
            </List.Item>
          )}
        />
      </div>
    );
  };

  const menuItems = [
    {
      key: 'analysis',
      icon: <SearchOutlined />,
      label: 'Analysis'
    },
    {
      key: 'history',
      icon: <HistoryOutlined />,
      label: 'History'
    },
    {
      key: 'saved',
      icon: <SaveOutlined />,
      label: 'Saved Analyses'
    }
  ];

  const processResponseToTree = (text: string): AnalysisNode[] => {
    const sections: AnalysisNode[] = [];
    let currentSection: AnalysisNode | null = null;
    let currentSubsection: AnalysisNode | null = null;
    
    // Split text into lines
    const lines = text.split('\n').filter(line => line.trim());
    
    lines.forEach((line, index) => {
      // Check for markdown headers
      if (line.startsWith('# ')) {
        currentSection = {
          key: `section-${index}`,
          title: line.replace('# ', ''),
          children: [],
          icon: <NodeIndexOutlined />
        };
        sections.push(currentSection);
      } else if (line.startsWith('## ')) {
        if (currentSection) {
          currentSubsection = {
            key: `subsection-${index}`,
            title: line.replace('## ', ''),
            children: [],
            icon: <BranchesOutlined />
          };
          currentSection.children?.push(currentSubsection);
        }
      } else if (line.startsWith('* ') || line.startsWith('- ')) {
        const node: AnalysisNode = {
          key: `point-${index}`,
          title: line.replace(/^[*-] /, ''),
          icon: <BulbOutlined />,
          content: line
        };
        
        if (currentSubsection) {
          currentSubsection.children = currentSubsection.children || [];
          currentSubsection.children.push(node);
        } else if (currentSection) {
          currentSection.children = currentSection.children || [];
          currentSection.children.push(node);
        } else {
          sections.push(node);
        }
      } else {
        // Regular text content
        const node: AnalysisNode = {
          key: `content-${index}`,
          title: line,
          icon: <SolutionOutlined />,
          content: line
        };
        
        if (currentSubsection) {
          currentSubsection.children = currentSubsection.children || [];
          currentSubsection.children.push(node);
        } else if (currentSection) {
          currentSection.children = currentSection.children || [];
          currentSection.children.push(node);
        } else {
          sections.push(node);
        }
      }
    });
    
    return sections;
  };

  const renderTreeNode = (node: AnalysisNode): TreeNodeData => {
    const title = (
      <Space>
        {node.icon}
        <span>{node.title}</span>
      </Space>
    );
    
    return {
      key: node.key,
      title: title,
      children: node.children?.map(child => renderTreeNode(child))
    };
  };

  const AnalysisTree: React.FC<{ content: string }> = ({ content }) => {
    const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([]);
    const treeData = processResponseToTree(content);
    
    return (
      <Card className="analysis-tree">
        <Tree
          showIcon
          defaultExpandAll
          expandedKeys={expandedKeys}
          onExpand={(keys) => setExpandedKeys(keys)}
          treeData={treeData.map(node => renderTreeNode(node))}
        />
      </Card>
    );
  };

  const handleViewChange = (view: ViewType) => {
    setActiveView(view);
    if (view === 'history') {
      loadHistory();
    }
  };

  const handleSubmit = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    
    const ragQuery: RAGQuery = {
      query: query.trim(),
      time_aware: timeAware,
      analysis_type: selectedAnalysisType
    };

    try {
      // Add loading message for better UX
      message.loading({
        content: 'Analyzing your query...',
        duration: 0,
        key: 'queryLoading'
      });
      
      const response = await submitRagQuery(ragQuery);
      
      // Clear loading message
      message.destroy('queryLoading');
      
      setResponse(response.response);
      
      // Use adapter to fix type issues
      const adaptedSources = Array.isArray(response.sources) 
        ? response.sources.map(adaptSourceToArticle) 
        : [];
      setSources(adaptedSources);
      
      setConfidence(response.confidence || 0);
      
      const historyItem: ExtendedRAGHistoryItem = {
        id: Date.now().toString(),
        query: ragQuery.query,
        response: response.response,
        timestamp: new Date().toISOString(),
        analysis_type: selectedAnalysisType,
        confidence: response.confidence,
        sources: adaptedSources  // Use adapted sources
      };
      
      setHistory(prev => [historyItem, ...prev]);
    } catch (err) {
      // Clear loading message
      message.destroy('queryLoading');
      
      console.error('Error submitting query:', err);
      
      // Create user-friendly error message
      let errorMessage = 'Failed to analyze your query. Please try again.';
      
      if (err instanceof Error) {
        // Check for network-related errors
        if (err.message.includes('Network error') || err.message.includes('Failed to fetch')) {
          errorMessage = 'Network error. Please check your connection and try again.';
        } else if (err.message.includes('timed out')) {
          errorMessage = 'Your query timed out. Please try a more specific question or break it into smaller parts.';
        } else if (err.message.includes('rate limit') || err.message.includes('quota')) {
          errorMessage = 'Rate limit exceeded. Please wait a few minutes and try again.';
        } else if (err.message.includes('Server error') || err.message.includes('unavailable')) {
          errorMessage = 'The AI service is temporarily unavailable. Please try again later.';
        } else {
          // Use the actual error message if available
          errorMessage = err.message;
        }
      }
      
      setError(errorMessage);
      
      // Use antd message for a less intrusive notification
      message.error({
        content: errorMessage,
        duration: 5,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSaveAnalysis = async () => {
    if (!response) return;
    
    try {
      setSavingAnalysis(true);
      
      const analysisData = {
        query: query,
        response: response,
        sources: sources,
        confidence: confidence,
        analysis_type: selectedAnalysisType,
        timestamp: new Date().toISOString()
      };
      
      const result = await saveAnalysis(analysisData);
      
      if (result.success) {
        message.success('Analysis saved successfully');
      } else {
        throw new Error('Failed to save analysis');
      }
      
    } catch (error) {
      console.error('Error saving analysis:', error);
      message.error('Failed to save analysis');
    } finally {
      setSavingAnalysis(false);
    }
  };

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <Content>
        <div className="container" style={{ maxWidth: 1200, margin: '0 auto', padding: '32px 0' }}>
          {/* Header card with improved styling */}
          <Card className="mb-6" style={{ marginBottom: 32, borderRadius: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={2} style={{ margin: 0, color: '#00405e', letterSpacing: 1 }}>
                  <Space>
                    <RocketOutlined style={{ fontSize: '32px', color: '#1890ff' }} />
                    KumbyAI Assistant
                  </Space>
                </Title>
                <Paragraph type="secondary" style={{ margin: '8px 0 0', color: '#7f9360', fontSize: 18 }}>
                  Analyze pharmaceutical data with AI-powered insights
                </Paragraph>
              </Col>
              <Col>
                <Space size="large">
                  <Tooltip title="Consider time relevance in analysis">
                    <Space>
                      <ClockCircleOutlined />
                      <Switch
                        checked={timeAware}
                        onChange={setTimeAware}
                      />
                      <Text type="secondary">Time-aware</Text>
                    </Space>
                  </Tooltip>
                </Space>
              </Col>
            </Row>
          </Card>
          
          {/* Navigation tabs with improved styling */}
          <Card className="mb-6" style={{ marginBottom: 32, borderRadius: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
            <Tabs 
              activeKey={activeView} 
              onChange={(key) => handleViewChange(key as ViewType)}
              size="large"
              type="card"
              tabBarStyle={{ marginBottom: 16, fontWeight: 600, fontSize: 18 }}
            >
              <TabPane 
                tab={<Space>{ANALYSIS_TYPES.analysis.icon} {ANALYSIS_TYPES.analysis.label}</Space>} 
                key="analysis"
              />
              <TabPane 
                tab={<Space>{ANALYSIS_TYPES.history.icon} {ANALYSIS_TYPES.history.label}</Space>} 
                key="history"
              />
              <TabPane 
                tab={<Space>{ANALYSIS_TYPES.saved.icon} {ANALYSIS_TYPES.saved.label}</Space>} 
                key="saved"
              />
            </Tabs>
          </Card>
          
          {/* Main content */}
          {activeView === 'analysis' ? (
            <Card className="analysis-card" style={{ borderRadius: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
              <Space direction="vertical" style={{ width: '100%' }} size="large">
                {/* Analysis Type Selector with improved styling */}
                <div>
                  <Title level={4} style={{ marginBottom: 12, color: '#00405e' }}>Select Analysis Type</Title>
                  <Select
                    style={{ width: '100%', fontSize: 16, borderRadius: 8 }}
                    value={selectedAnalysisType}
                    onChange={handleAnalysisTypeChange}
                    disabled={loading}
                    optionLabelProp="label"
                    size="large"
                    dropdownRender={menu => (
                      <div>
                        <div style={{ padding: '8px 12px', color: 'rgba(0, 0, 0, 0.45)', fontWeight: 'bold', backgroundColor: '#f5f5f5' }}>
                          Choose Analysis Method
                        </div>
                        {menu}
                      </div>
                    )}
                  >
                    {Object.entries(CATEGORIES).map(([category, types]) => (
                      <Select.OptGroup key={category} label={
                        <Text strong style={{ color: '#7f9360' }}>{category}</Text>
                      }>
                        {types.map(({ key, label, icon, description }) => (
                          <Option key={key} value={key} label={
                            <Space>
                              {icon}
                              <Text>{label}</Text>
                            </Space>
                          }>
                            <Space align="start">
                              {icon}
                              <div>
                                <Text strong>{label}</Text>
                                <br />
                                <Text type="secondary" style={{ fontSize: '12px' }}>
                                  {description}
                                </Text>
                              </div>
                            </Space>
                          </Option>
                        ))}
                      </Select.OptGroup>
                    ))}
                  </Select>
                </div>

                {/* Query Input with improved styling */}
                <div>
                  <Title level={4} style={{ marginBottom: 12, color: '#00405e' }}>Your Question</Title>
                  <TextArea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={ANALYSIS_TYPES[selectedAnalysisType].description}
                    autoSize={{ minRows: 3, maxRows: 6 }}
                    disabled={loading}
                    onPressEnter={(e) => {
                      if (!e.shiftKey) {
                        e.preventDefault();
                        handleStreamQuery();
                      }
                    }}
                    style={{ 
                      fontSize: '16px', 
                      borderRadius: '8px',
                      padding: '16px',
                      boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.08)',
                      border: '1.5px solid #d9e3e0',
                      background: '#f8fafc'
                    }}
                  />
                  <div style={{ 
                    marginTop: '16px', 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center' 
                  }}>
                    {confidence > 0 && renderConfidenceIndicator(confidence)}
                    <Space>
                      <Button
                        onClick={() => setQuery('')}
                        icon={<ClearOutlined />}
                        disabled={!query || loading}
                        style={{ borderRadius: 8 }}
                      >
                        Clear
                      </Button>
                      <Button
                        type="primary"
                        onClick={handleStreamQuery}
                        icon={<SendOutlined />}
                        loading={loading}
                        disabled={!query.trim()}
                        size="large"
                        style={{ borderRadius: 8, fontWeight: 600, letterSpacing: 0.5 }}
                      >
                        {loading ? 'Analyzing...' : 'Analyze'}
                      </Button>
                    </Space>
                  </div>
                </div>

                {/* Error and Response Display */}
                {error && (
                  <Alert
                    message="Analysis Error"
                    description={error}
                    type="error"
                    showIcon
                    closable
                    onClose={() => setError(null)}
                    style={{ marginBottom: 16 }}
                  />
                )}

                {response && (
                  <div className="analysis-results">
                    <Divider orientation="left">
                      <Space>
                        {ANALYSIS_TYPES[selectedAnalysisType].icon}
                        <Text strong style={{ fontSize: '18px' }}>Analysis Results</Text>
                        <Button 
                          type="primary"
                          icon={<SaveOutlined />} 
                          onClick={handleSaveAnalysis}
                          loading={savingAnalysis}
                          style={{ borderRadius: 8 }}
                        >
                          Save Analysis
                        </Button>
                      </Space>
                    </Divider>
                    <Row gutter={[24, 24]}>
                      <Col xs={24} md={8}>
                        <Card 
                          title={<Space>{<BranchesOutlined />} Content Structure</Space>} 
                          className="tree-card"
                          style={{ height: '100%', borderRadius: 12, background: '#f6f8fa' }}
                        >
                          <AnalysisTree content={response} />
                        </Card>
                      </Col>
                      <Col xs={24} md={16}>
                        <Card 
                          title={<Space>{<SolutionOutlined />} Detailed Analysis</Space>}
                          className="markdown-preview"
                          style={{ height: '100%', borderRadius: 12, background: '#fff' }}
                        >
                          <div 
                            className="markdown-content" 
                            style={{ 
                              fontSize: '16px', 
                              lineHeight: '1.7',
                              padding: '20px',
                              backgroundColor: '#fafafa',
                              borderRadius: '8px',
                              minHeight: 200
                            }}
                          >
                            <ReactMarkdown>{response}</ReactMarkdown>
                          </div>
                        </Card>
                      </Col>
                    </Row>
                    
                    {/* Sources with improved styling */}
                    {sources && sources.length > 0 && (
                      <div style={{ marginTop: '24px' }}>
                        <Card
                          title={
                            <Space>
                              <InfoCircleOutlined />
                              <Text strong>Sources ({sources.length})</Text>
                            </Space>
                          }
                          className="sources-card"
                          style={{ borderRadius: 12, background: '#f6f8fa' }}
                        >
                          <List
                            size="small"
                            dataSource={sources}
                            renderItem={(source, index) => (
                              <List.Item>
                                <Space direction="vertical" style={{ width: '100%' }}>
                                  <Space>
                                    <Badge 
                                      count={index + 1} 
                                      style={{ 
                                        backgroundColor: '#1890ff',
                                        fontSize: '12px',
                                        minWidth: '22px',
                                        height: '22px',
                                        lineHeight: '22px'
                                      }} 
                                    />
                                    <Tag color="blue">{source.source}</Tag>
                                    {source.date && (
                                      <Tag color="green">
                                        {(() => {
                                          try {
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
                                      </Tag>
                                    )}
                                  </Space>
                                  <a 
                                    href={source.link} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    style={{ fontWeight: 'bold' }}
                                  >
                                    {source.title}
                                  </a>
                                </Space>
                              </List.Item>
                            )}
                          />
                        </Card>
                      </div>
                    )}
                    <div ref={responseEndRef} />
                  </div>
                )}
              </Space>
            </Card>
          ) : activeView === 'history' ? (
            <HistoryView
              activeView={activeView}
              setActiveView={setActiveView}
              history={history}
              handleClearHistory={handleClearHistory}
              loadHistoryItem={loadHistoryItem}
            />
          ) : (
            <SavedAnalysisView
              activeView={activeView}
              setActiveView={setActiveView}
              loadSavedAnalysis={loadSavedAnalysis}
            />
          )}
        </div>
      </Content>
    </Layout>
  );
};

export default RAGPage;

// Add styles at the end of the file
const styles = `
  /* General styles */
  .container {
    transition: all 0.3s ease;
  }
  
  /* Tree styling */
  .analysis-tree .ant-tree-node-content-wrapper {
    white-space: normal;
    height: auto;
    padding: 8px 0;
  }
  
  .analysis-tree .ant-tree-title {
    display: inline-block;
    width: 100%;
  }
  
  .tree-card .ant-card-body {
    max-height: 600px;
    overflow-y: auto;
  }
  
  /* Markdown content */
  .markdown-preview {
    background: #fff;
  }
  
  .markdown-content {
    font-size: 16px;
    line-height: 1.6;
    max-height: 600px;
    overflow-y: auto;
  }
  
  .markdown-content h1 {
    font-size: 1.8em;
    margin-top: 0.5em;
    margin-bottom: 0.5em;
    color: #1890ff;
  }
  
  .markdown-content h2 {
    font-size: 1.5em;
    margin-top: 0.75em;
    margin-bottom: 0.5em;
    padding-bottom: 0.25em;
    border-bottom: 1px solid #f0f0f0;
    color: #333;
  }
  
  .markdown-content h3 {
    font-size: 1.3em;
    margin-top: 0.75em;
    margin-bottom: 0.5em;
    color: #333;
  }
  
  .markdown-content p {
    margin-bottom: 1em;
  }
  
  .markdown-content ul, .markdown-content ol {
    padding-left: 1.5em;
    margin-bottom: 1em;
  }
  
  .markdown-content li {
    margin-bottom: 0.5em;
  }
  
  .markdown-content blockquote {
    border-left: 4px solid #1890ff;
    padding-left: 1em;
    margin-left: 0;
    margin-right: 0;
    color: #666;
  }
  
  .markdown-content code {
    background: #f5f5f5;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-size: 0.9em;
  }
  
  .markdown-content pre {
    background: #f5f5f5;
    padding: 1em;
    border-radius: 6px;
    overflow-x: auto;
  }
  
  /* List views */
  .history-item, .saved-item {
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  }
  
  .history-item:hover, .saved-item:hover {
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    border-color: #d9d9d9;
  }
  
  /* Headers */
  .history-header, .saved-header {
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  /* Sources section */
  .sources-card .ant-list-item {
    padding: 12px;
    border-radius: 4px;
    margin-bottom: 8px;
    transition: all 0.3s ease;
  }
  
  .sources-card .ant-list-item:hover {
    background-color: #f5f5f5;
  }
  
  /* Confidence indicator */
  .confidence-indicator {
    border-radius: 12px;
    padding: 4px 8px;
    font-size: 12px;
    font-weight: bold;
  }
`;

// Add style tag to document
if (typeof document !== 'undefined') {
  const styleTag = document.createElement('style');
  styleTag.innerHTML = styles;
  document.head.appendChild(styleTag);
} 