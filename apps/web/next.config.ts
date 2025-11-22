import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  experimental: {
    clientSegmentCache: true
  },
  transpilePackages: ['@3dix/ui', '@3dix/three', '@3dix/types']
};

export default nextConfig;
