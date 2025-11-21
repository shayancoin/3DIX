import React from 'react';
import * as THREE from 'three';

export type SurfaceType = 'floor' | 'wall' | 'ceiling';

export type SurfaceOrientation = 'back' | 'front' | 'left' | 'right' | 'top' | 'bottom';

export interface SurfaceProps {
  type: SurfaceType;
  orientation?: SurfaceOrientation;
  width: number;
  height: number;
  position?: [number, number, number];
  rotation?: [number, number, number];
  color?: string;
  opacity?: number;
  textureUrl?: string;
  receiveShadow?: boolean;
  castShadow?: boolean;
  doubleSided?: boolean;
  material?: 'standard' | 'basic' | 'phong';
}

/**
 * Surface component for rendering room surfaces (floor, walls, ceiling)
 * Provides a reusable abstraction for rendering planes in 3D scenes
 */
export function Surface({
  type,
  orientation,
  width,
  height,
  position = [0, 0, 0],
  rotation,
  color = '#F5F5F5',
  opacity = 0.3,
  textureUrl,
  receiveShadow = true,
  castShadow = false,
  doubleSided = false,
  material = 'standard',
}: SurfaceProps) {
  // Determine rotation based on type and orientation if not explicitly provided
  let finalRotation: [number, number, number] = rotation || [0, 0, 0];

  if (!rotation) {
    switch (type) {
      case 'floor':
        finalRotation = [-Math.PI / 2, 0, 0];
        break;
      case 'ceiling':
        finalRotation = [Math.PI / 2, 0, 0];
        break;
      case 'wall':
        if (orientation === 'back') {
          finalRotation = [0, 0, 0];
        } else if (orientation === 'front') {
          finalRotation = [0, Math.PI, 0];
        } else if (orientation === 'left') {
          finalRotation = [0, Math.PI / 2, 0];
        } else if (orientation === 'right') {
          finalRotation = [0, -Math.PI / 2, 0];
        }
        break;
    }
  }

  // Create material props
  const materialProps: any = {
    color,
    opacity,
    transparent: opacity < 1,
    side: doubleSided ? THREE.DoubleSide : THREE.FrontSide,
  };

  // Load texture if provided
  const [texture, setTexture] = React.useState<THREE.Texture | null>(null);

  React.useEffect(() => {
    if (textureUrl) {
      const loader = new THREE.TextureLoader();
      loader.load(
        textureUrl,
        (loadedTexture) => {
          setTexture(loadedTexture);
        },
        undefined,
        (error) => {
          console.warn('Failed to load texture:', error);
        }
      );
    }
  }, [textureUrl]);

  // Render material based on type
  const renderMaterial = () => {
    const props: any = { ...materialProps };
    if (texture) {
      props.map = texture;
    }

    switch (material) {
      case 'basic':
        return <meshBasicMaterial {...props} />;
      case 'phong':
        return <meshPhongMaterial {...props} />;
      case 'standard':
      default:
        return <meshStandardMaterial {...props} />;
    }
  };

  return (
    <mesh
      position={position}
      rotation={finalRotation}
      receiveShadow={receiveShadow}
      castShadow={castShadow}
    >
      <planeGeometry args={[width, height]} />
      {renderMaterial()}
    </mesh>
  );
}

/**
 * RoomSurfaces component for rendering all surfaces of a room
 */
export interface RoomSurfacesProps {
  width: number;
  length: number;
  height: number;
  floorColor?: string;
  wallColor?: string;
  ceilingColor?: string;
  floorOpacity?: number;
  wallOpacity?: number;
  ceilingOpacity?: number;
  showFloor?: boolean;
  showWalls?: boolean;
  showCeiling?: boolean;
  floorTextureUrl?: string;
  wallTextureUrl?: string;
  ceilingTextureUrl?: string;
  receiveShadow?: boolean;
}

export function RoomSurfaces({
  width,
  length,
  height,
  floorColor = '#F5F5F5',
  wallColor = '#E0E0E0',
  ceilingColor = '#FFFFFF',
  floorOpacity = 0.3,
  wallOpacity = 0.2,
  ceilingOpacity = 0.1,
  showFloor = true,
  showWalls = true,
  showCeiling = false,
  floorTextureUrl,
  wallTextureUrl,
  ceilingTextureUrl,
  receiveShadow = true,
}: RoomSurfacesProps) {
  return (
    <group>
      {/* Floor */}
      {showFloor && (
        <Surface
          type="floor"
          width={width}
          height={length}
          position={[width / 2, 0, length / 2]}
          color={floorColor}
          opacity={floorOpacity}
          textureUrl={floorTextureUrl}
          receiveShadow={receiveShadow}
        />
      )}

      {/* Walls */}
      {showWalls && (
        <>
          {/* Back wall */}
          <Surface
            type="wall"
            orientation="back"
            width={width}
            height={height}
            position={[width / 2, height / 2, 0]}
            color={wallColor}
            opacity={wallOpacity}
            textureUrl={wallTextureUrl}
            receiveShadow={receiveShadow}
            doubleSided
          />
          {/* Left wall */}
          <Surface
            type="wall"
            orientation="left"
            width={length}
            height={height}
            position={[0, height / 2, length / 2]}
            color={wallColor}
            opacity={wallOpacity}
            textureUrl={wallTextureUrl}
            receiveShadow={receiveShadow}
            doubleSided
          />
          {/* Right wall */}
          <Surface
            type="wall"
            orientation="right"
            width={length}
            height={height}
            position={[width, height / 2, length / 2]}
            color={wallColor}
            opacity={wallOpacity}
            textureUrl={wallTextureUrl}
            receiveShadow={receiveShadow}
            doubleSided
          />
          {/* Front wall */}
          <Surface
            type="wall"
            orientation="front"
            width={width}
            height={height}
            position={[width / 2, height / 2, length]}
            color={wallColor}
            opacity={wallOpacity}
            textureUrl={wallTextureUrl}
            receiveShadow={receiveShadow}
            doubleSided
          />
        </>
      )}

      {/* Ceiling */}
      {showCeiling && (
        <Surface
          type="ceiling"
          width={width}
          height={length}
          position={[width / 2, height, length / 2]}
          color={ceilingColor}
          opacity={ceilingOpacity}
          textureUrl={ceilingTextureUrl}
          receiveShadow={receiveShadow}
        />
      )}
    </group>
  );
}
