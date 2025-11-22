'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Stage, Layer, Rect, Group, Text, Transformer } from 'react-konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { SceneObject2D, CanvasViewport, Point2D } from '@3dix/types';

interface Canvas2DProps {
  width: number;
  height: number;
  objects: SceneObject2D[];
  viewport?: CanvasViewport;
  onObjectsChange?: (objects: SceneObject2D[]) => void;
  onViewportChange?: (viewport: CanvasViewport) => void;
  selectedObjectId?: string;
  onObjectSelect?: (objectId: string | undefined) => void;
  roomWidth?: number; // Room dimensions in meters
  roomLength?: number;
}

// Category color mapping
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
  default: '#6C757D',
};

export function Canvas2D({
  width,
  height,
  objects,
  viewport = { x: 0, y: 0, zoom: 1 },
  onObjectsChange,
  onViewportChange,
  selectedObjectId,
  onObjectSelect,
  roomWidth = 5,
  roomLength = 4,
}: Canvas2DProps) {
  const stageRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<Point2D | null>(null);

  // Scale factor: pixels per meter
  const scale = Math.min(width / roomWidth, height / roomLength) * 0.8;
  const offsetX = (width - roomWidth * scale) / 2;
  const offsetY = (height - roomLength * scale) / 2;

  // Update transformer when selection changes
  useEffect(() => {
    if (transformerRef.current && selectedObjectId) {
      const selectedNode = stageRef.current?.findOne(`#${selectedObjectId}`);
      if (selectedNode) {
        transformerRef.current.nodes([selectedNode]);
        transformerRef.current.getLayer()?.batchDraw();
      }
    } else if (transformerRef.current) {
      transformerRef.current.nodes([]);
    }
  }, [selectedObjectId]);

  const handleObjectClick = useCallback(
    (e: KonvaEventObject<MouseEvent>, objectId: string) => {
      e.cancelBubble = true;
      onObjectSelect?.(objectId === selectedObjectId ? undefined : objectId);
    },
    [selectedObjectId, onObjectSelect]
  );

  const handleObjectDragStart = useCallback(
    (e: KonvaEventObject<DragEvent>, objectId: string) => {
      setIsDragging(true);
      const obj = objects.find((o) => o.id === objectId);
      if (obj) {
        setDragStart({ x: obj.position.x, y: obj.position.y });
      }
    },
    [objects]
  );

  const handleObjectDragEnd = useCallback(
    (e: KonvaEventObject<DragEvent>, objectId: string) => {
      setIsDragging(false);
      const node = e.target;
      const newX = (node.x() - offsetX) / scale;
      const newY = (node.y() - offsetY) / scale;

      const updatedObjects = objects.map((obj) =>
        obj.id === objectId
          ? {
            ...obj,
            position: { x: newX, y: newY },
          }
          : obj
      );

      onObjectsChange?.(updatedObjects);
      setDragStart(null);
    },
    [objects, onObjectsChange, offsetX, offsetY, scale]
  );

  const handleStageClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      const clickedOnEmpty = e.target === e.target.getStage();
      if (clickedOnEmpty) {
        onObjectSelect?.(undefined);
      }
    },
    [onObjectSelect]
  );

  const handleWheel = useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      e.evt.preventDefault();
      const stage = e.target.getStage();
      if (!stage) return;

      const oldScale = viewport.zoom;
      const pointer = stage.getPointerPosition();
      if (!pointer) return;

      const mousePointTo = {
        x: (pointer.x - viewport.x) / oldScale,
        y: (pointer.y - viewport.y) / oldScale,
      };

      const newScale = e.evt.deltaY > 0 ? oldScale * 0.95 : oldScale * 1.05;
      const clampedScale = Math.max(0.5, Math.min(3, newScale));

      const newViewport = {
        x: pointer.x - mousePointTo.x * clampedScale,
        y: pointer.y - mousePointTo.y * clampedScale,
        zoom: clampedScale,
      };

      onViewportChange?.(newViewport);
    },
    [viewport, onViewportChange]
  );

  return (
    <div className="relative w-full h-full bg-gray-100 border border-gray-300 rounded-lg overflow-hidden">
      <Stage
        ref={stageRef}
        width={width}
        height={height}
        onClick={handleStageClick}
        onWheel={handleWheel}
        style={{ cursor: isDragging ? 'grabbing' : 'default' }}
      >
        <Layer>
          {/* Room boundary */}
          <Rect
            x={offsetX}
            y={offsetY}
            width={roomWidth * scale}
            height={roomLength * scale}
            fill="#F5F5F5"
            stroke="#CCCCCC"
            strokeWidth={2}
            dash={[5, 5]}
          />

          {/* Grid */}
          {Array.from({ length: Math.ceil(roomWidth) + 1 }).map((_, i) => (
            <React.Fragment key={`grid-v-${i}`}>
              <Rect
                x={offsetX + i * scale}
                y={offsetY}
                width={1}
                height={roomLength * scale}
                fill="#E0E0E0"
                opacity={0.3}
              />
            </React.Fragment>
          ))}
          {Array.from({ length: Math.ceil(roomLength) + 1 }).map((_, i) => (
            <React.Fragment key={`grid-h-${i}`}>
              <Rect
                x={offsetX}
                y={offsetY + i * scale}
                width={roomWidth * scale}
                height={1}
                fill="#E0E0E0"
                opacity={0.3}
              />
            </React.Fragment>
          ))}

          {/* Scene objects */}
          {objects.map((obj) => {
            const x = offsetX + obj.position.x * scale;
            const y = offsetY + obj.position.y * scale;
            const objWidth = obj.dimensions.width * scale;
            const objHeight = obj.dimensions.depth * scale;
            const isSelected = obj.id === selectedObjectId;
            const color = obj.color || CATEGORY_COLORS[obj.category] || CATEGORY_COLORS.default;

            return (
              <Group
                key={obj.id}
                id={obj.id}
                x={x}
                y={y}
                rotation={obj.rotation || 0}
                draggable
                onDragStart={(e) => handleObjectDragStart(e, obj.id)}
                onDragEnd={(e) => handleObjectDragEnd(e, obj.id)}
                onClick={(e) => handleObjectClick(e, obj.id)}
              >
                <Rect
                  width={objWidth}
                  height={objHeight}
                  fill={color}
                  opacity={0.7}
                  stroke={isSelected ? '#3B82F6' : '#000000'}
                  strokeWidth={isSelected ? 3 : 1}
                  shadowBlur={isSelected ? 10 : 0}
                  shadowColor="#3B82F6"
                />
                {obj.label && (
                  <Text
                    text={obj.label}
                    fontSize={12}
                    fill="#000000"
                    x={5}
                    y={5}
                    width={objWidth - 10}
                    align="left"
                    wrap="word"
                  />
                )}
                <Text
                  text={obj.category}
                  fontSize={10}
                  fill="#666666"
                  x={5}
                  y={objHeight - 15}
                  width={objWidth - 10}
                  align="left"
                />
              </Group>
            );
          })}

          {/* Transformer for selected object */}
          <Transformer
            ref={transformerRef}
            boundBoxFunc={(oldBox, newBox) => {
              // Limit resize to keep aspect ratio reasonable
              if (Math.abs(newBox.width) < 5 || Math.abs(newBox.height) < 5) {
                return oldBox;
              }
              return newBox;
            }}
          />
        </Layer>
      </Stage>
    </div>
  );
}
