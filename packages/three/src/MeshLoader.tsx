import React, { Suspense, useState, useEffect } from 'react';
import { useGLTF } from '@react-three/drei';
import { Mesh, Group, BoxGeometry, MeshStandardMaterial } from 'three';
import { LayoutObject } from '@3dix/types';

interface MeshLoaderProps {
  object: LayoutObject;
  quality?: 'low' | 'medium' | 'high';
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

function ModelMesh({ url, onLoad, onError }: { url: string; onLoad?: () => void; onError?: (error: Error) => void }) {
  const gltf = useGLTF(url);
  
  useEffect(() => {
    if (gltf?.scene && onLoad) {
      onLoad();
    }
  }, [gltf, onLoad]);

  if (!gltf?.scene) {
    return null;
  }

  return <primitive object={gltf.scene} />;
}

function FallbackBox({ size, color }: { size: [number, number, number]; color: string }) {
  return (
    <mesh>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

export function MeshLoader({ object, quality = 'high', onLoad, onError }: MeshLoaderProps) {
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    // Get asset URL from metadata
    const assetUrl = object.metadata?.assetUrl;
    if (!assetUrl) {
      setError(new Error('No asset URL in object metadata'));
      setLoading(false);
      return;
    }

    // Adjust URL based on quality
    let url = assetUrl;
    if (quality === 'low' && assetUrl.includes('/model.gltf')) {
      url = assetUrl.replace('/model.gltf', '/model_low.gltf');
    } else if (quality === 'medium' && assetUrl.includes('/model.gltf')) {
      url = assetUrl.replace('/model.gltf', '/model_medium.gltf');
    }

    setMeshUrl(url);
    setLoading(true);
    setError(null);
  }, [object, quality]);

  const [width, height, depth] = object.size;
  const color = object.metadata?.color || '#6C757D';

  if (error || !meshUrl) {
    return <FallbackBox size={[width, height, depth]} color={color} />;
  }

  return (
    <Suspense fallback={<FallbackBox size={[width, height, depth]} color={color} />}>
      <ModelMesh
        url={meshUrl}
        onLoad={() => {
          setLoading(false);
          onLoad?.();
        }}
        onError={(err) => {
          setError(err);
          setLoading(false);
          onError?.(err);
        }}
      />
    </Suspense>
  );
}

// Preload GLTF files
useGLTF.preload = (url: string) => {
  // This will be handled by drei's useGLTF
};
