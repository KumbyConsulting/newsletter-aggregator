import type { NextConfig } from "next";

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  experimental: {
    esmExternals: true,
    // Enable module resolution features
    serverComponentsExternalPackages: ['lodash']
  },
  webpack: (config, { isServer }) => {
    // Add custom webpack config for module resolution
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': require('path').resolve(__dirname, './src'),
    };
    return config;
  },
  async rewrites() {
    // Prioritize API Gateway URL, fallback to direct API URL, then localhost
    const apiGatewayUrl = process.env.NEXT_PUBLIC_API_GATEWAY_URL;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
    
    const targetUrl = apiGatewayUrl || apiUrl;
    console.log('API Target URL:', targetUrl); // Debug log
    
    return [
      {
        source: '/api/:path*',
        destination: `${targetUrl}/api/:path*`,
      },
      {
        source: '/ws/:path*',
        destination: `${targetUrl}/ws/:path*`,
      }
    ];
  },
  // Add env configuration
  env: {
    NEXT_PUBLIC_API_GATEWAY_URL: process.env.NEXT_PUBLIC_API_GATEWAY_URL,
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
};

export default nextConfig;
