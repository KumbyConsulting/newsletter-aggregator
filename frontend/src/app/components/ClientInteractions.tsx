'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Button, notification, Space, Alert, Progress, Typography, App, message as staticMessage } from 'antd';
import { SyncOutlined, CheckCircleOutlined, CloseCircleOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { startUpdate, getUpdateStatus, pollUpdateStatus, clearCache } from '@/app/services/api';
import { UpdateStatus, ApiErrorResponse } from '@/types';
import React from 'react';

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
                  const response = await fetch('/api/update/cancel', { method: 'POST' });
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
  const pollingInterval = useRef<NodeJS.Timeout | null>(null);
  const retryCount = useRef(0);
  const maxRetries = 3;
  const basePollingInterval = 3000; // Start with 3 seconds
  const maxPollingInterval = 15000; // Max 15 seconds

  // Adaptive polling interval based on progress
  const getPollingInterval = (progress: number) => {
    if (progress < 25) return basePollingInterval;
    if (progress < 50) return basePollingInterval * 1.5;
    if (progress < 75) return basePollingInterval * 2;
    return maxPollingInterval;
  };

  // Function to check status with retry logic
  const checkStatus = useCallback(async (initialCheck = false) => {
    try {
      const status = await getUpdateStatus();
      retryCount.current = 0; // Reset retry count on successful call
      setUpdateStatus(status);

      if (status.in_progress) {
        setShowUpdateStatus(true);
        setIsUpdating(true);
        
        // Adjust polling interval based on progress
        if (!pollingInterval.current) {
          startPolling(status.progress);
        }
      } else {
        setIsUpdating(false);
        if (!initialCheck && (status.status === 'completed' || status.status === 'failed' || status.status.startsWith('completed_with'))) {
          setShowUpdateStatus(true);
          // Keep error states visible until manually closed
          if (status.status !== 'failed' && !status.status.includes('error')) {
            const timeout = status.status.includes('warning') ? 10000 : 5000;
            setTimeout(() => setShowUpdateStatus(false), timeout);
          }
          
          if (status.status === 'completed' || status.status.startsWith('completed_with')) {
            console.log('Update finished, clearing API cache...');
            clearCache();
          }
        }
        stopPolling();
      }
    } catch (error) {
      console.error('Error fetching update status:', error);
      const apiError = error as ApiErrorResponse;
      
      // Implement retry logic for transient failures
      if (retryCount.current < maxRetries) {
        retryCount.current++;
        const retryDelay = Math.min(1000 * Math.pow(2, retryCount.current), 8000);
        setTimeout(() => checkStatus(initialCheck), retryDelay);
        return;
      }

      message.error(apiError.error || 'Failed to check update status');
      setIsUpdating(false);
      stopPolling();
    }
  }, [message]);

  // Enhanced polling logic with adaptive intervals
  const startPolling = (progress: number) => {
    if (pollingInterval.current) return;
    const interval = getPollingInterval(progress);
    pollingInterval.current = setInterval(() => checkStatus(), interval);
  };

  const stopPolling = () => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
  };

  // Check status on initial mount
  useEffect(() => {
    checkStatus(true);
    // Cleanup polling on unmount
    return () => stopPolling();
  }, [checkStatus]);

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
      startPolling(0); // Start with base polling interval
    } catch (error) {
      console.error('Error starting update:', error);
      const apiError = error as ApiErrorResponse;
      
      // Enhanced error handling with specific messages
      const errorMessage = apiError.error || 'Failed to start update.';
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

      {/* Update Button */}
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