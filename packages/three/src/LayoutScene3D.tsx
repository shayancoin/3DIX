import React, { useCallback, useMemo, useRef, useState } from 'react';
import { Canvas, ThreeEvent, useFrame } from '@react-three/fiber';
import { Grid, OrbitControls, Text, useGLTF } from '@react-three/drei';
import { SceneObject3D } from '@3dix/types';
import * as THREE from 'three';

export type LayoutScene3DProps = {
  objects: SceneObject3D[];
  roomOutline?: [number, number][];
  selectedId?: string;
  onSelect?: (id: string | null) => void;
  quality?: 'low' | 'medium' | 'high';
};

const CATEGORY_COLORS: Record<string, string> = {
  sofa: '#20B2AA',
  table: '#D2691E',
  chair: '#9370DB',
  bed: '#FF69B4',
  default: '#6C757D',
};

function ObjectMesh({
  object,
  selected,
  onClick,
  quality,
}: {
  object: SceneObject3D;
  selected: boolean;
  onClick: (e: ThreeEvent<MouseEvent>) => void;
  quality: 'low' | 'medium' | 'high';
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  useFrame((state) => {
    if (meshRef.current && selected) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime) * 0.1;
    }
  });

  const color = CATEGORY_COLORS[object.category] || CATEGORY_COLORS.default;
  const [x, y, z] = object.position;
  const [w, h, d] = object.size;
  const rotationY = (object.orientation || 0) * (Math.PI / 2);

  return (
    <group
      position={[x, y + h / 2, z]}
      rotation={[0, rotationY, 0]}
      onClick={onClick}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <mesh ref={meshRef}>
        <boxGeometry args={[w, h, d]} />
        <meshStandardMaterial
          color={selected ? '#3B82F6' : color}
          opacity={hovered || selected ? 0.9 : 0.7}
          transparent
          emissive={selected ? '#3B82F6' : '#000000'}
          emissiveIntensity={selected ? 0.3 : 0}
          visible={quality === 'low' || !object.mesh_url}
        />
      </mesh>
      {selected && (
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(w * 1.05, h * 1.05, d * 1.05)]} />
          <lineBasicMaterial color="#3B82F6" />
        </lineSegments>
      )}
      {quality !== 'low' && object.mesh_url && (
        <AssetMesh meshUrl={object.mesh_url} targetSize={[w, h, d]} />
      )}
      <Text position={[0, h / 2 + 0.15, 0]} fontSize={0.18} color="#111">
        {object.category}
      </Text>
    </group>
  );
}

function AssetMesh({ meshUrl, targetSize }: { meshUrl: string; targetSize: [number, number, number] }) {
  const gltf = useGLTF(meshUrl, true);
  const scene = useMemo(() => gltf.scene.clone(true), [gltf.scene]);

  // Fit the mesh to target size
  useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);
    const [w, h, d] = targetSize;
    const scale = new THREE.Vector3(
      size.x ? w / size.x : 1,
      size.y ? h / size.y : 1,
      size.z ? d / size.z : 1
    );
    scene.scale.copy(scale);
    scene.position.sub(center);
    scene.position.y += size.y / 2;
  }, [scene, targetSize]);

  return <primitive object={scene} />;
}
useGLTF.preload('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb');

function SceneContents({
  objects,
  roomOutline,
  selectedId,
  onSelect,
  quality = 'high',
}: LayoutScene3DProps) {
  const bounds = useMemo(() => {
    if (!roomOutline || roomOutline.length === 0) {
      return { minX: 0, maxX: 5, minZ: 0, maxZ: 4 };
    }
    const xs = roomOutline.map((p) => p[0]);
    const zs = roomOutline.map((p) => p[1]);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minZ: Math.min(...zs),
      maxZ: Math.max(...zs),
    };
  }, [roomOutline]);

  const width = bounds.maxX - bounds.minX || 5;
  const length = bounds.maxZ - bounds.minZ || 4;
  const center: [number, number, number] = [
    bounds.minX + width / 2,
    0,
    bounds.minZ + length / 2,
  ];

  const handleBackgroundClick = useCallback(() => {
    onSelect?.(null);
  }, [onSelect]);

  return (
    <>
      <OrbitControls target={center} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[8, 10, 6]} intensity={0.9} />
      <Grid
        infiniteGrid
        cellSize={0.5}
        cellThickness={0.4}
        sectionSize={2}
        sectionThickness={1}
        fadeDistance={20}
        fadeStrength={1}
      />
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[center[0], 0, center[2]]}
        onClick={handleBackgroundClick}
      >
        <planeGeometry args={[width || 5, length || 4]} />
        <meshStandardMaterial color="#f8fafc" opacity={0.4} transparent />
      </mesh>

      {objects.map((obj) => (
        <ObjectMesh
          key={obj.id}
          object={obj}
          selected={obj.id === selectedId}
          quality={quality}
          onClick={(e) => {
            e.stopPropagation();
            onSelect?.(obj.id === selectedId ? null : obj.id);
          }}
        />
      ))}
    </>
  );
}

export function LayoutScene3D(props: LayoutScene3DProps) {
  const { objects } = props;
  if (!objects || objects.length === 0) {
    return (
      <div className="w-full h-full bg-slate-100 flex items-center justify-center text-sm text-muted-foreground">
        No objects to render
      </div>
    );
  }

  return (
    <Canvas shadows camera={{ position: [8, 6, 8], fov: 50 }}>
      <SceneContents {...props} />
    </Canvas>
  );
}
