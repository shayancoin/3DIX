import React, { useRef, useState, useCallback } from 'react';
import { useFrame, ThreeEvent } from '@react-three/fiber';
import { Box, Grid, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { LayoutObject, Point3D } from '@3dix/types';
import * as THREE from 'three';
import { MeshLoader } from './MeshLoader';
import { RoomSurfaces } from './Surface';

interface LayoutScene3DProps {
  objects: LayoutObject[];
  roomWidth?: number;
  roomLength?: number;
  roomHeight?: number;
  selectedObjectId?: string;
  onObjectClick?: (objectId: string | undefined) => void;
  cameraPosition?: [number, number, number];
  showGrid?: boolean;
  showLabels?: boolean;
  meshQuality?: 'low' | 'medium' | 'high';
  useMeshes?: boolean;
}

// Category color mapping (matching 2D canvas)
const CATEGORY_COLORS: Record<string, string> = {
  refrigerator: '#4A90E2',
  sink: '#50C878',
  cabinet: '#8B4513',
  stove: '#FF6B6B',
  dishwasher: '#FFD93D',
  counter: '#A0A0A0',
  table: '#D2691E',
  chair: '#9370DB',
  bed: '#FF69B4',
  sofa: '#20B2AA',
  toilet: '#87CEEB',
  shower: '#4682B4',
  dresser: '#CD853F',
  nightstand: '#DDA0DD',
  default: '#6C757D',
};

function LayoutObject3D({
  object,
  isSelected,
  onClick,
  useMesh,
  meshQuality,
}: {
  object: LayoutObject;
  isSelected: boolean;
  onClick: (e: ThreeEvent<MouseEvent>) => void;
  useMesh?: boolean;
  meshQuality?: 'low' | 'medium' | 'high';
}) {
  const meshRef = useRef<THREE.Mesh | THREE.Group>(null);
  const [hovered, setHovered] = useState(false);

  // Animate selected objects
  useFrame((state) => {
    if (meshRef.current && isSelected) {
      if ('rotation' in meshRef.current) {
        meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime) * 0.1;
      }
    }
  });

  const color = object.metadata?.color || CATEGORY_COLORS[object.category] || CATEGORY_COLORS.default;
  const [x, y, z] = object.position;
  const [width, height, depth] = object.size;
  const hasAsset = (object.metadata?.assetUrl || object.metadata?.isCustom) && useMesh;

  return (
    <group
      position={[x, y + height / 2, z]}
      rotation={[0, object.orientation, 0]}
      onClick={onClick}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {hasAsset ? (
        <group ref={meshRef as any}>
          <MeshLoader object={object} quality={meshQuality} />
          {/* Highlight overlay for selected objects */}
          {isSelected && (
            <mesh>
              <boxGeometry args={[width * 1.05, height * 1.05, depth * 1.05]} />
              <meshStandardMaterial
                color="#3B82F6"
                opacity={0.2}
                transparent
                wireframe
              />
            </mesh>
          )}
        </group>
      ) : (
        <mesh ref={meshRef as any}>
          <boxGeometry args={[width, height, depth]} />
          <meshStandardMaterial
            color={isSelected ? '#3B82F6' : hovered ? color : color}
            opacity={isSelected ? 0.9 : 0.7}
            transparent
            emissive={isSelected ? '#3B82F6' : '#000000'}
            emissiveIntensity={isSelected ? 0.3 : 0}
          />
        </mesh>
      )}
      {/* Wireframe outline for selected objects */}
      {isSelected && !hasAsset && (
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(width, height, depth)]} />
          <lineBasicMaterial color="#3B82F6" linewidth={3} />
        </lineSegments>
      )}
      {/* Label */}
      {object.category && (
        <Text
          position={[0, height / 2 + 0.2, 0]}
          fontSize={0.2}
          color="#000000"
          anchorX="center"
          anchorY="middle"
        >
          {object.category}
        </Text>
      )}
    </group>
  );
}

export function LayoutScene3D({
  objects,
  roomWidth = 5,
  roomLength = 4,
  roomHeight = 2.5,
  selectedObjectId,
  onObjectClick,
  cameraPosition = [8, 6, 8],
  showGrid = true,
  showLabels = true,
  meshQuality = 'high',
  useMeshes = true,
}: LayoutScene3DProps) {
  const handleObjectClick = useCallback(
    (e: ThreeEvent<MouseEvent>, objectId: string) => {
      e.stopPropagation();
      onObjectClick?.(objectId === selectedObjectId ? undefined : objectId);
    },
    [selectedObjectId, onObjectClick]
  );

  const handleBackgroundClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      if (e.object === e.eventObject) {
        onObjectClick?.(undefined);
      }
    },
    [onObjectClick]
  );

  return (
    <>
      {/* Camera */}
      <PerspectiveCamera makeDefault position={cameraPosition} fov={50} />
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={3}
        maxDistance={20}
        target={[roomWidth / 2, 0, roomLength / 2]}
      />

      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      <directionalLight position={[-10, 10, -5]} intensity={0.5} />

      {/* Grid */}
      {showGrid && (
        <Grid
          infiniteGrid
          cellSize={1}
          cellThickness={0.5}
          cellColor="#E0E0E0"
          sectionSize={5}
          sectionThickness={1}
          sectionColor="#CCCCCC"
          fadeDistance={25}
          fadeStrength={1}
        />
      )}

      {/* Room boundary visualization */}
      <group onClick={handleBackgroundClick}>
        <RoomSurfaces
          width={roomWidth}
          length={roomLength}
          height={roomHeight}
          showFloor={true}
          showWalls={true}
          showCeiling={false}
          receiveShadow={true}
        />
      </group>

      {/* Layout Objects */}
      {objects.map((obj) => (
        <LayoutObject3D
          key={obj.id}
          object={obj}
          isSelected={obj.id === selectedObjectId}
          onClick={(e) => handleObjectClick(e, obj.id)}
          useMesh={useMeshes}
          meshQuality={meshQuality}
        />
      ))}

      {/* Axes helper (optional, for debugging) */}
      <axesHelper args={[2]} />
    </>
  );
}
