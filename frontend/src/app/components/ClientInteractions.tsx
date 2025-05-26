'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Button, notification, Space, Alert, Progress, Typography, App, message as staticMessage } from 'antd';
import { SyncOutlined, CheckCircleOutlined, CloseCircleOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { startUpdate, getUpdateStatus, pollUpdateStatus, clearCache } from '@/services/api';
import { UpdateStatus, ApiErrorResponse } from '@/types';
import React from 'react';
import { signInWithGoogle, onAuthStateChangedHelper, getIdToken } from '@/utils/firebaseClient';

const { Text } = Typography;

// Placeholder for the actual status display component
const UpdateStatusIndicator = ({ status, onClose }: { status: UpdateStatus | null, onClose: () => void }) => {
  if (!status || !status.in_progress) return null;

  let alertType: "info" | "success" | "error" | "warning" = "info";
  let icon = <SyncOutlined spin />;

  if (status.status === 'completed') {
    alertType = 'success';
    icon = <CheckCircleOutlined />;
  } else if (status.status === 'failed') {
    alertType = 'error';
    icon = <CloseCircleOutlined />;
  } else if (status.status === 'completed_with_errors' || status.status === 'completed_with_warnings') {
    alertType = 'warning';
    icon = <InfoCircleOutlined />;
  }

  // Calculate estimated time remaining
  const timeRemaining = status.estimated_completion_time 
    ? Math.max(0, Math.round((status.estimated_completion_time - Date.now()) / 1000))
    : null;

  return (
    <Alert
      message={
        <Space>
          {icon}
          <Text strong>Update Status: {status.message || status.status}</Text>
          {status.can_be_cancelled && (
            <Button 
              size="small" 
              danger
              onClick={async () => {
                try {
                  // Get Firebase ID token
                  const token = await getIdToken();
                  if (!token) {
                    throw new Error('You must be logged in to cancel updates.');
                  }
                  const response = await fetch('/api/update/cancel', {
                    method: 'POST',
                    headers: {
                      'Authorization': `Bearer ${token}`,
                      'Content-Type': 'application/json'
                    }
                  });
                  const data = await response.json();
                  if (!data.success) {
                    throw new Error(data.message);
                  }
                } catch (error) {
                  console.error('Error cancelling update:', error);
                }
              }}
            >
              Cancel
            </Button>
          )}
        </Space>
      }
      description={
        <Space direction="vertical" style={{ width: '100%' }}>
          {status.in_progress && status.status !== 'completed' && status.status !== 'failed' && (
            <>
              <Progress 
                percent={status.progress} 
                size="small" 
                status={alertType === 'error' ? 'exception' : 'active'} 
              />
              {timeRemaining !== null && (
                <Text type="secondary">
                  Estimated time remaining: {timeRemaining > 60 
                    ? `${Math.round(timeRemaining / 60)} minutes` 
                    : `${timeRemaining} seconds`}
                </Text>
              )}
            </>
          )}
          {status.error && <Text type="danger">Error: {status.error}</Text>}
          {status.total_sources > 0 && (
            <Text type="secondary">
              Processed {status.sources_processed}/{status.total_sources} sources | Found {status.articles_found} articles
            </Text>
          )}
        </Space>
      }
      type={alertType}
      showIcon
      closable
      onClose={onClose}
      style={{ marginBottom: 16 }}
    />
  );
};

// Internal component wrapped with App context
function UpdateButton() {
  const { message } = App.useApp();
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus | null>(null);
  const [showUpdateStatus, setShowUpdateStatus] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [user, setUser] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Track auth state
  useEffect(() => {
    const unsubscribe = onAuthStateChangedHelper(setUser);
    return () => unsubscribe();
  }, []);

  // Helper to start polling
  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return;
    pollIntervalRef.current = setInterval(async () => {
      try {
        const status = await getUpdateStatus();
        setUpdateStatus(status);
        if (status.in_progress) {
          setShowUpdateStatus(true);
          setIsUpdating(true);
        } else {
          setIsUpdating(false);
        }
      } catch (e) {
        // Optionally handle polling error
      }
    }, 3000);
  }, []);

  // Helper to stop polling
  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  // WebSocket connection for update status, with polling fallback
  useEffect(() => {
    const connectWS = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      // Use env variable or default to current host for prod/dev
      const backendHost = process.env.NEXT_PUBLIC_BACKEND_WS_HOST || window.location.host;
      const wsUrl = `${protocol}://${backendHost}/ws/status`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setUpdateStatus(data);
          if (data.in_progress) {
            setShowUpdateStatus(true);
            setIsUpdating(true);
          } else {
            setIsUpdating(false);
          }
        } catch (e) {}
      };
      ws.onerror = () => {
        ws.close();
      };
      ws.onclose = () => {
        wsRef.current = null;
        // Start polling if WebSocket closes
        startPolling();
        if (!reconnectTimeout.current) {
          reconnectTimeout.current = setTimeout(() => {
            reconnectTimeout.current = null;
            stopPolling(); // Stop polling if WS reconnects
            connectWS();
          }, 2000);
        }
      };
    };
    connectWS();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }
      stopPolling();
    };
  }, [startPolling, stopPolling]);

  // Handle the update button click
  const handleUpdateClick = async () => {
    if (isUpdating) {
      message.info('Update already in progress.');
      return;
    }
    
    setIsUpdating(true);
    setShowUpdateStatus(true);
    setUpdateStatus(prev => ({ 
      ...(prev ?? {} as UpdateStatus), 
      status: 'starting', 
      message: 'Initiating update...', 
      progress: 0, 
      in_progress: true 
    }));

    try {
      const result = await startUpdate();
      setUpdateStatus(result.status);
      message.success(result.message || 'Update process started successfully');
    } catch (error) {
      console.error('Error starting update:', error);
      
      // More robust error extraction for different API response formats
      let errorMessage = 'Failed to start update.';
      
      if (typeof error === 'object' && error !== null) {
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        
        // Try to extract error from different response formats
        const errorObj = error as Record<string, any>;
        if (errorObj.apiError && typeof errorObj.apiError === 'object') {
          errorMessage = errorObj.apiError.error || errorObj.apiError.message || errorMessage;
        } else if ('error' in errorObj && typeof errorObj.error === 'string') {
          errorMessage = errorObj.error || errorMessage;
        } else if ('message' in errorObj && typeof errorObj.message === 'string') {
          errorMessage = errorObj.message;
        } else if ('statusText' in errorObj && typeof errorObj.statusText === 'string') {
          errorMessage = `Server error: ${errorObj.statusText}`;
        }
      }
      
      message.error({
        content: errorMessage,
        duration: 5,
        onClick: () => message.destroy()
      });

      setUpdateStatus(prev => ({ 
        ...(prev ?? {} as UpdateStatus), 
        status: 'failed', 
        message: 'Failed to start', 
        error: errorMessage, 
        in_progress: false 
      }));
      setIsUpdating(false);
      setShowUpdateStatus(true);
    }
  };

  return (
    <div style={{ position: 'fixed', top: 80, right: 20, zIndex: 1000 }}>
      {/* Display the Update Status Indicator */}
      {showUpdateStatus && (
        <UpdateStatusIndicator
          status={updateStatus}
          onClose={() => setShowUpdateStatus(false)}
        />
      )}

      {/* Update Button: Only show if user is logged in */}
      {user && (
        <Button
          type="primary"
          icon={<SyncOutlined spin={isUpdating} />}
          onClick={handleUpdateClick}
          disabled={isUpdating}
          size="large"
          shape="circle"
          aria-label="Update Articles"
          style={{
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
          }}
        />
      )}
    </div>
  );
}

// Export the component wrapped with App context
export default function ClientInteractions() {
  return (
    <App>
      <UpdateButton />
    </App>
  );
} 