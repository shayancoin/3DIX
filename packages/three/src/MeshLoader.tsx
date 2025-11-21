import React, { Suspense, useState, useEffect } from 'react';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { LayoutObject } from '@3dix/types';

interface MeshLoaderProps {
  object: LayoutObject;
  quality?: 'low' | 'medium' | 'high';
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

function ModelMesh({ 
  url, 
  targetSize, 
  onLoad, 
  onError 
}: { 
  url: string; 
  targetSize: [number, number, number];
  onLoad?: () => void; 
  onError?: (error: Error) => void;
}) {
  const gltf = useGLTF(url);
  const meshRef = React.useRef<THREE.Group>(null);
  
  useEffect(() => {
    if (gltf?.scene && onLoad) {
      onLoad();
    }
  }, [gltf, onLoad]);

  // Scale and center the mesh to match target size
  useEffect(() => {
    if (gltf?.scene && meshRef.current) {
      const scene = gltf.scene.clone();
      
      // Calculate bounding box
      const box = new THREE.Box3().setFromObject(scene);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      
      // Calculate scale factors for each dimension
      const scaleX = targetSize[0] / size.x;
      const scaleY = targetSize[1] / size.y;
      const scaleZ = targetSize[2] / size.z;
      
      // Use uniform scaling to maintain aspect ratio, or non-uniform if needed
      // For furniture, we typically want to match the target size closely
      const uniformScale = Math.min(scaleX, scaleY, scaleZ);
      
      // Apply scaling
      scene.scale.set(uniformScale, uniformScale, uniformScale);
      
      // Center the mesh
      scene.position.sub(center.multiplyScalar(uniformScale));
      
      // Update the ref
      if (meshRef.current) {
        meshRef.current.clear();
        meshRef.current.add(scene);
      }
    }
  }, [gltf, targetSize]);

  if (!gltf?.scene) {
    return null;
  }

  return <group ref={meshRef} />;
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
  const [customMesh, setCustomMesh] = useState<Group | null>(null);

  // Handle custom mesh data (base64 encoded)
  useEffect(() => {
    if (object.metadata?.customMeshData && object.metadata?.isCustom) {
      setLoading(true);
      try {
        // Decode base64 mesh data
        const meshData = atob(object.metadata.customMeshData);
        // For PLY format, create a data URL and load it
        // Note: This is a simplified approach. In production, use a proper PLY loader
        const blob = new Blob([meshData], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        
        // For now, we'll use a placeholder. In production, integrate PLYLoader
        // TODO: Integrate THREE.PLYLoader or similar to load PLY files
        console.log('Custom mesh data available, but PLY loading requires additional loader');
        setCustomMesh(null);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load custom mesh:', err);
        setError(err instanceof Error ? err : new Error('Failed to load custom mesh'));
        setLoading(false);
      }
    }
  }, [object.metadata?.customMeshData, object.metadata?.isCustom]);

  useEffect(() => {
    // Skip if custom mesh
    if (object.metadata?.isCustom) {
      return;
    }

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
  const color = object.metadata?.isCustom ? '#4CAF50' : (object.metadata?.color || '#6C757D');

  // Render custom mesh if available (placeholder for now)
  if (object.metadata?.isCustom) {
    if (customMesh) {
      return <primitive object={customMesh.clone()} />;
    }
    // Fallback to colored box for custom objects
    return <FallbackBox size={[width, height, depth]} color={color} />;
  }

  if (error || !meshUrl) {
    return <FallbackBox size={[width, height, depth]} color={color} />;
  }

  return (
    <Suspense fallback={<FallbackBox size={[width, height, depth]} color={color} />}>
      <ModelMesh
        url={meshUrl}
        targetSize={[width, height, depth]}
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
