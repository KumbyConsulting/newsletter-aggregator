import { Alert, Typography, Button, Space } from 'antd';

const { Text } = Typography;

interface ErrorDisplayProps {
  error: string | { message?: string; error?: string } | any;
  statusCode?: number;
  onRetry?: () => void;
}

export default function ErrorDisplay({ error, statusCode, onRetry }: ErrorDisplayProps) {
  const title = statusCode ? `Error ${statusCode}` : 'Error';
  
  // Extract error message from various possible formats
  let errorMessage: string;
  if (typeof error === 'string') {
    errorMessage = error;
  } else if (error && typeof error === 'object') {
    errorMessage = error.message || error.error || JSON.stringify(error);
  } else {
    errorMessage = 'An unknown error occurred.';
  }
  
  // Format user-friendly message based on error type
  let userFriendlyMessage = errorMessage;
  let showRetry = false;
  
  if (errorMessage.includes('timed out')) {
    userFriendlyMessage = 'The request timed out. The server might be busy or slow. Please try again in a moment.';
    showRetry = true;
  } else if (errorMessage.includes('Network') || errorMessage.includes('Failed to fetch')) {
    userFriendlyMessage = 'Network error. Please check your internet connection and try again.';
    showRetry = true;
  } else if (statusCode === 500 || errorMessage.includes('500')) {
    userFriendlyMessage = 'The server encountered an internal error. Our team has been notified.';
  } else if (statusCode === 404 || errorMessage.includes('404')) {
    userFriendlyMessage = 'The requested resource was not found.';
  } else if (errorMessage.includes('unavailable') || errorMessage.includes('Server error')) {
    userFriendlyMessage = 'The service is temporarily unavailable. Please try again later.';
    showRetry = true;
  } else if (statusCode === 502 || statusCode === 503 || statusCode === 504) {
    userFriendlyMessage = 'The backend service is temporarily unavailable. This might indicate a deployment mismatch between frontend and backend services.';
    showRetry = true;
  }

  return (
    <Alert
      message={title}
      description={
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text>{userFriendlyMessage}</Text>
          {process.env.NODE_ENV === 'development' && <Text type="secondary" style={{ fontSize: '12px' }}>Raw error: {errorMessage}</Text>}
          {showRetry && onRetry && (
            <Button type="primary" onClick={onRetry} size="small">
              Retry
            </Button>
          )}
        </Space>
      }
      type="error"
      showIcon
    />
  );
} 