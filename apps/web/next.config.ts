import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  experimental: {
    ppr: true,
    clientSegmentCache: true,
    nodeMiddleware: true
  },
  transpilePackages: ['@3dix/ui', '@3dix/three']
};

export default nextConfig;
