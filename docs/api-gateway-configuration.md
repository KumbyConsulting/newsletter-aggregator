# API Gateway Configuration Documentation

## Overview
This document outlines the current API Gateway configuration for the Newsletter Aggregator project. The configuration was examined on April 21, 2025.

## Configuration Components

### 1. API Configuration
```bash
# Command used:
gcloud api-gateway apis list
gcloud api-gateway apis describe newsletter-aggregator-api
```

**Details:**
- **API ID:** newsletter-aggregator-api
- **Display Name:** newsletter-aggregator-api
- **State:** ACTIVE
- **Managed Service:** newsletter-aggregator-api-1v5ua4g6l66k7.apigateway.newsletter-450510.cloud.goog
- **Creation Time:** 2025-04-19T20:48:47
- **Last Update:** 2025-04-19T20:50:25

### 2. Gateway Configuration
```bash
# Commands used:
gcloud api-gateway gateways list
gcloud api-gateway gateways describe newsletter-aggregator-gateway --location=us-central1
```

**Details:**
- **Gateway ID:** newsletter-aggregator-gateway
- **Location:** us-central1
- **Display Name:** newsletter-aggregator-gateway
- **State:** ACTIVE
- **Default Hostname:** newsletter-aggregator-gateway-axs105xr.uc.gateway.dev
- **Creation Time:** 2025-04-19T20:53:13
- **Last Update:** 2025-04-21T22:35:11
- **API Config:** newsletter-aggregator-config-v10 (Updated from v2)

### 3. API Configuration Versions
The API has multiple configuration versions, with v10 being the currently active version:

Available Versions (from newest to oldest):
- newsletter-aggregator-config-v11 (2025-04-21T21:36:00) - Changed service account, not used
- newsletter-aggregator-config-v10 (2025-04-21T12:21:33) - Currently active
- newsletter-aggregator-config-v9 (2025-04-21T09:27:53)
- newsletter-aggregator-config-v8 (2025-04-21T09:03:22)
- newsletter-aggregator-config-v2 (2025-04-21T22:32:00) - Previously active
- newsletter-aggregator-config-v1 (2025-04-21T20:34:18)

### 4. Frontend Integration Requirements
The frontend service has specific requirements that the API Gateway must satisfy:

**Environment Support:**
```typescript
// Allowed Origins (CORS)
- https://newsletter-aggregator-knap.vercel.app
- https://newsletter-aggregator.vercel.app
- http://localhost:3000

// API Gateway Configuration
const API_GATEWAY_URL = process.env.NEXT_PUBLIC_API_GATEWAY_URL || 'https://newsletter-aggregator-gateway-axs105xr.uc.gateway.dev';
const API_BASE_URL = `${API_GATEWAY_URL}/api`;
```

**Security Requirements:**
- API Key Authentication (x-api-key header)
- CORS with credentials support
- Secure HTTPS endpoints

**Performance Features:**
- Request timeouts:
  - Standard endpoints: 10 seconds
  - Long-running operations: 30 seconds
- Caching implementation:
  - Articles: 60 seconds
  - Topics: 300 seconds
  - Search results: 30 seconds
- Retry mechanism with exponential backoff
- Request deduplication

### 5. OpenAPI Specification Analysis
The OpenAPI specification (openapi.yaml) defines:

**Backend Service:**
- Host: newsletter-aggregator-ukm23f55ra-uc.a.run.app
- Protocol: h2 (HTTP/2)

**Security:**
- API Key authentication required
- CORS configuration matches frontend requirements
- Credentials allowed
- Maximum CORS age: 3600 seconds

**Endpoints:**
- All required frontend endpoints are properly mapped
- Timeout configurations are appropriate for each endpoint type
- Error responses are properly defined

## Current Status
- Gateway using v10 configuration (stable version)
- All components in ACTIVE state
- Deployed in us-central1 region
- Frontend requirements fully satisfied
- CORS properly configured
- Service account authentication configured correctly

## Security Considerations
- API Gateway uses a dedicated service account for authentication
- Gateway is publicly accessible but requires API key authentication
- All configurations are in an ACTIVE state
- CORS is properly restricted to allowed origins
- Timeouts are configured to prevent hanging connections

## Recent Changes
- Reverted from v2 to v10 due to stability issues
- Avoided v11 due to service account changes
- Current configuration (v10) provides stable operation

## Monitoring Recommendations
1. Set up alerts for:
   - 4xx errors (client errors)
   - 5xx errors (server errors)
   - Latency spikes
   - Failed requests
   - Rate limiting events

2. Monitor:
   - API usage patterns
   - Cache hit rates
   - Error rates by endpoint
   - Response times
   - CORS issues

## Next Steps
- [ ] Set up detailed monitoring for the v10 configuration
- [ ] Document specific differences between v10 and v11 configurations
- [ ] Implement rate limiting monitoring
- [ ] Set up alerting for critical endpoints
- [ ] Create a rollback plan if needed 