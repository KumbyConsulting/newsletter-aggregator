import { Alert, Typography } from 'antd';

const { Text } = Typography;

interface ErrorDisplayProps {
  error: string;
  statusCode?: number;
}

export default function ErrorDisplay({ error, statusCode }: ErrorDisplayProps) {
  const title = statusCode ? `Error ${statusCode}` : 'Error';

  return (
    <Alert
      message={title}
      description={<Text>{error || 'An unknown error occurred.'}</Text>}
      type="error"
      showIcon
    />
  );
} 