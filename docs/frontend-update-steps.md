# Frontend Update Steps for API Gateway Changes

## Overview
This document outlines the steps required to update the Vercel-deployed frontend application to work with the revised API Gateway configuration (v10) and Cloud Run backend service.

## Backend Service Information
```yaml
Backend Service Details:
- Service Name: newsletter-aggregator
- URL: https://newsletter-aggregator-ukm23f55ra-uc.a.run.app
- Region: us-central1
- Service Account: newsletter-aggregator-sa@newsletter-450510.iam.gserviceaccount.com
- Container: gcr.io/newsletter-450510/newsletter-aggregator:v1
- Resources:
  - CPU: 2000m (2 cores)
  - Memory: 4Gi
- Scaling:
  - Min Instances: 1
  - Max Instances: 10
  - Container Concurrency: 80
```

## 1. Environment Variables Update

### Current Production Environment Variables
```bash
# Vercel Environment Variables
NEXT_PUBLIC_API_GATEWAY_URL=https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev
NEXT_PUBLIC_API_KEY=[your-api-key]

# Additional Backend Configuration
INTERNAL_BACKEND_URL=https://newsletter-aggregator-ukm23f55ra-uc.a.run.app
```

### Steps to Update in Vercel Dashboard
1. Navigate to the Vercel Dashboard
2. Select the newsletter-aggregator project
3. Go to Settings > Environment Variables
4. Verify/Update the following variables:
   - `NEXT_PUBLIC_API_GATEWAY_URL`
   - `NEXT_PUBLIC_API_KEY`
   - `INTERNAL_BACKEND_URL`
5. If using different environments (Preview/Development), update those accordingly

## 2. Deployment Configuration

### Update vercel.json
```json
{
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Credentials", "value": "true" },
        { "key": "Access-Control-Allow-Origin", "value": "https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev" },
        { "key": "Access-Control-Allow-Methods", "value": "GET,DELETE,PATCH,POST,PUT,OPTIONS" },
        { "key": "Access-Control-Allow-Headers", "value": "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, x-api-key" }
      ]
    }
  ],
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://newsletter-aggregator-ukm23f55ra-uc.a.run.app/api/:path*"
    }
  ]
}
```

## 3. Cache Configuration Updates

Update the cache settings in `frontend/src/services/api.ts`:

```typescript
// Article cache TTLs (milliseconds)
const CACHE_TTLS = {
  ARTICLES: 60000,      // 1 minute
  TOPICS: 300000,       // 5 minutes
  TOPICS_STATS: 300000, // 5 minutes
  SEARCH: 30000,        // 30 seconds
  SIMILAR: 300000,      // 5 minutes
};

// Update timeout configurations to match backend
const TIMEOUT_CONFIG = {
  DEFAULT: 30000,      // 30 seconds
  LONG_RUNNING: 600000 // 10 minutes (matching Cloud Run timeout)
};
```

## 4. Backend Health Check

Add a health check before deployment:
```typescript
async function checkBackendHealth() {
  try {
    const response = await fetch(`${INTERNAL_BACKEND_URL}/health`, {
      timeout: 5000
    });
    return response.ok;
  } catch (error) {
    console.error('Backend health check failed:', error);
    return false;
  }
}
```

## 5. Deployment Steps

1. **Pre-deployment Verification**
   ```bash
   # Verify backend service is running
   curl https://newsletter-aggregator-ukm23f55ra-uc.a.run.app/health
   
   # Verify API Gateway
   curl -H "x-api-key: $API_KEY" https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev/api/health
   
   # Verify current branch
   git checkout main
   
   # Pull latest changes
   git pull origin main
   ```

2. **Update Dependencies**
   ```bash
   # Install dependencies
   npm install
   
   # Update API-related packages if needed
   npm update @types/api-gateway
   
   # Build locally to verify
   npm run build
   ```

3. **Deploy to Vercel**
   ```bash
   # Using Vercel CLI
   vercel --prod
   
   # Or push to main branch if you have auto-deployment configured
   git push origin main
   ```

## 6. Testing Checklist

- [ ] Test API Gateway connectivity
- [ ] Verify CORS configuration
- [ ] Check authentication flow
- [ ] Test caching behavior
- [ ] Verify error handling
- [ ] Test rate limiting
- [ ] Check streaming endpoints
- [ ] Verify long-running operations
- [ ] Test backend service direct connection
- [ ] Verify resource limits (memory/CPU)
- [ ] Check autoscaling behavior

## 7. Monitoring Setup

1. **Add Error Monitoring**
   ```typescript
   // Add to _app.tsx or similar
   if (typeof window !== 'undefined') {
     window.onerror = function(msg, url, lineNo, columnNo, error) {
       console.error('Frontend Error:', {
         message: msg,
         url: url,
         line: lineNo,
         column: columnNo,
         error: error
       });
       // Add your error reporting service here
       return false;
     };
   }
   ```

2. **Add API Request Monitoring**
   ```typescript
   // Add to api.ts
   const monitorApiRequest = async (endpoint: string, startTime: number) => {
     const duration = Date.now() - startTime;
     console.log(`API Request to ${endpoint} took ${duration}ms`);
     
     // Monitor backend service limits
     if (duration > TIMEOUT_CONFIG.DEFAULT) {
       console.warn(`Request to ${endpoint} exceeded standard timeout`);
     }
   };
   ```

## 8. Rollback Plan

If issues are encountered:

1. **Quick Rollback**
   ```bash
   # Using Vercel CLI
   vercel rollback
   ```

2. **Manual Rollback Steps**
   - Revert to previous environment variables
   - Redeploy using previous configuration
   - Monitor error rates during rollback
   - Verify backend service connectivity

## 9. Post-Deployment Verification

1. **Check Critical Endpoints**
   ```typescript
   const criticalEndpoints = [
     '/api/articles',
     '/api/topics',
     '/api/update/status',
     '/api/rag'
   ];

   async function verifyEndpoints() {
     for (const endpoint of criticalEndpoints) {
       const startTime = Date.now();
       try {
         const response = await fetch(endpoint);
         const duration = Date.now() - startTime;
         
         console.log(`${endpoint}: ${response.status} (${duration}ms)`);
         console.assert(response.ok, `${endpoint} is not responding correctly`);
         
         // Check response time against backend limits
         if (duration > 5000) {
           console.warn(`${endpoint} response time exceeds 5s threshold`);
         }
       } catch (error) {
         console.error(`${endpoint} check failed:`, error);
       }
     }
   }
   ```

2. **Verify Performance**
   - Check response times against Cloud Run limits
   - Monitor cache hit rates
   - Verify error rates
   - Check API Gateway metrics
   - Monitor backend service scaling

## Contact Information

For deployment issues or questions:
- DevOps Team: [contact information]
- API Gateway Team: [contact information]
- Frontend Team: [contact information]
- Backend Team: [contact information]

## Additional Notes

- Backend service is configured with a 10-minute timeout (600s)
- Maximum concurrent requests per instance: 80
- Service will scale between 1 and 10 instances
- Monitor memory usage (4GB limit per instance)
- Keep monitoring the deployment for at least 24 hours
- Document any issues encountered during the update process
- Update the team on the deployment status 