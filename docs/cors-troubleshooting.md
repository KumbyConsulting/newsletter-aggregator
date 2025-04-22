# CORS Troubleshooting Guide

## Current Error
```
Access to fetch at 'https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev/api/topics/stats' 
from origin 'https://newsletter-aggregator-knap.vercel.app' has been blocked by CORS policy
```

## Immediate Fix

1. **Update API Gateway Configuration**
```bash
# Create a new configuration version with updated CORS
gcloud api-gateway api-configs create newsletter-aggregator-config-v12 \
  --api=newsletter-aggregator-api \
  --openapi-spec=openapi.yaml \
  --project=newsletter-450510

# Update the gateway to use the new configuration
gcloud api-gateway gateways update newsletter-aggregator-gateway \
  --api=newsletter-aggregator-api \
  --api-config=newsletter-aggregator-config-v12 \
  --location=us-central1
```

2. **Frontend Temporary Workaround**
Update `frontend/src/services/api.ts`:

```typescript
async function apiRequest<T>(
  endpoint: string, 
  options: RequestInit = {}, 
  cacheKey: string = '', 
  cacheTTL: number = 60000,
  timeoutMs: number = 30000,
): Promise<T> {
  const url = new URL(`${API_BASE_URL}${endpoint}`);
  
  // Enhanced headers for CORS
  const headers = {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY,
    'Accept': 'application/json',
    'Origin': window.location.origin,
    ...options.headers,
  };

  const requestOptions = {
    ...options,
    headers,
    mode: 'cors' as RequestMode,
    credentials: 'omit' as RequestCredentials,
  };

  // Add retry logic for CORS errors
  return retryWithBackoff(
    async () => {
      try {
        const response = await fetch(url.toString(), requestOptions);
        if (!response.ok) {
          if (response.status === 0) {
            throw new Error('CORS error - check API Gateway configuration');
          }
          // ... rest of error handling
        }
        return response.json();
      } catch (error) {
        console.error('Request failed:', error);
        throw error;
      }
    },
    3, // retries
    1000, // initialDelay
    10000, // maxDelay
    2 // factor
  );
}
```

3. **Vercel Configuration**
Update `vercel.json`:

```json
{
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Credentials", "value": "true" },
        { "key": "Access-Control-Allow-Origin", "value": "https://newsletter-aggregator-knap.vercel.app" },
        { "key": "Access-Control-Allow-Methods", "value": "GET,DELETE,PATCH,POST,PUT,OPTIONS" },
        { "key": "Access-Control-Allow-Headers", "value": "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, x-api-key, Origin" }
      ]
    }
  ]
}
```

## Verification Steps

1. **Test API Gateway CORS Configuration**
```bash
# Test preflight request
curl -X OPTIONS -H "Origin: https://newsletter-aggregator-knap.vercel.app" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: x-api-key" \
  -v https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev/api/topics/stats

# Test actual request
curl -H "Origin: https://newsletter-aggregator-knap.vercel.app" \
  -H "x-api-key: $API_KEY" \
  https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev/api/topics/stats
```

2. **Browser Test**
```javascript
// Run in browser console
fetch('https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev/api/topics/stats', {
  headers: {
    'x-api-key': 'your-api-key',
  },
  mode: 'cors'
}).then(r => r.json()).then(console.log).catch(console.error);
```

## Long-term Solutions

1. **Use Cloud Run URL Mapping**
```yaml
# In Cloud Run service configuration
metadata:
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/allowed-origins: https://newsletter-aggregator-knap.vercel.app
```

2. **Implement API Gateway Caching**
```yaml
# In OpenAPI spec
x-google-backend:
  address: https://newsletter-aggregator-ukm23f55ra-uc.a.run.app
  protocol: h2
  cache_control:
    maxAge: 300s
```

3. **Set Up Error Monitoring**
```typescript
// Add to frontend
window.addEventListener('unhandledrejection', function(event) {
  if (event.reason instanceof Error) {
    if (event.reason.message.includes('CORS')) {
      console.error('CORS Error:', {
        url: event.reason.stack,
        origin: window.location.origin,
        timestamp: new Date().toISOString()
      });
    }
  }
});
```

## Common Issues and Solutions

1. **Missing CORS Headers**
- Verify API Gateway configuration
- Check Cloud Run service headers
- Validate Vercel configuration

2. **Preflight Failures**
- Ensure OPTIONS method is allowed
- Check allowed headers configuration
- Verify allowed origins list

3. **Authentication Issues**
- Confirm API key header is in allowed headers
- Check API key validity
- Verify request includes correct credentials mode

4. **Cache Issues**
- Clear browser cache
- Check API Gateway cache settings
- Verify Cloud Run cache headers

## Monitoring

1. **Add CORS Error Tracking**
```typescript
// Add to api.ts
const trackCorsError = (endpoint: string, error: Error) => {
  console.error('CORS Error:', {
    endpoint,
    error: error.message,
    origin: window.location.origin,
    timestamp: new Date().toISOString()
  });
};
```

2. **Monitor API Gateway Metrics**
```bash
# View CORS-related metrics
gcloud api-gateway gateways describe newsletter-aggregator-gateway \
  --location=us-central1 \
  --format='yaml(metrics)'
``` 