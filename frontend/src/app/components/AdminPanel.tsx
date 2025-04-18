import React, { useState } from 'react';
import { Button, Card, Divider, Modal, Space, Alert } from 'antd';
import { ApiOutlined } from '@ant-design/icons';
import { message } from 'antd';

const AdminPanel: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div>
      {/* Add this after other sections in the admin panel */}
      <Divider orientation="left">AI Service Diagnostics</Divider>
      <Card title="AI Model Connection Status" className="mb-4">
        <Space direction="vertical" style={{ width: '100%' }}>
          <Button 
            type="primary" 
            icon={<ApiOutlined />}
            onClick={async () => {
              try {
                setIsLoading(true);
                const response = await fetch('/api/diagnostics/ai-check');
                
                if (!response.ok) {
                  throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Format the diagnostic data nicely
                const diagnosticInfo = `
AI Service Diagnostics:
---------------------
Connection status: ${data.connection_status}
Test result: ${data.test_result}
${data.error ? `Error: ${data.error}` : ''}

API Key: ${data.api_key_status?.present ? 'Present' : 'Missing'}
${data.api_key_status?.masked ? `Key: ${data.api_key_status.masked}` : ''}

${data.test_response ? `
Test Response:
Prompt: "${data.test_response.prompt}"
Response: "${data.test_response.response}"
Response length: ${data.test_response.response_length} characters
` : 'No test response available'}

Timestamp: ${data.timestamp}
                `;
                
                // Show modal with diagnostic info
                Modal.info({
                  title: 'AI Service Diagnostics',
                  content: (
                    <div>
                      <pre style={{ whiteSpace: 'pre-wrap', maxHeight: '400px', overflow: 'auto' }}>
                        {diagnosticInfo}
                      </pre>
                    </div>
                  ),
                  width: 600,
                });
                
              } catch (error) {
                message.error(`Failed to check AI service: ${error instanceof Error ? error.message : String(error)}`);
              } finally {
                setIsLoading(false);
              }
            }}
          >
            Check AI Service Status
          </Button>
          
          <Alert
            type="info"
            message="AI Service Troubleshooting"
            description={
              <ul>
                <li>If the AI service is not responding, check the API key configuration</li>
                <li>Empty responses may indicate rate limiting or quota issues</li>
                <li>For connection issues, verify network connectivity to the AI provider</li>
                <li>Consistent timeouts may indicate content that's too large for processing</li>
              </ul>
            }
          />
        </Space>
      </Card>
    </div>
  );
};

export default AdminPanel; 