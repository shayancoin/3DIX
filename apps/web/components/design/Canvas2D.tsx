'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Rect, Line, Group, Text } from 'react-konva';
import { SceneObject2D } from '@3dix/types';
import useImage from 'use-image';

interface Canvas2DProps {
    width: number;
    height: number;
    roomDimensions: { width: number; length: number }; // in meters
    objects: SceneObject2D[];
    onObjectUpdate: (id: string, newProps: Partial<SceneObject2D>) => void;
    selectedObjectId?: string | null;
    onSelectObject?: (id: string | null) => void;
}

const GRID_SIZE = 50; // pixels per meter
const STAGE_PADDING = 50;

/**
 * Render a pannable, zoomable 2D room canvas with a grid, room boundary, and draggable scene objects.
 *
 * @param width - Stage width in pixels
 * @param height - Stage height in pixels
 * @param roomDimensions - Room size in meters (`width` and `length`) used to compute pixel dimensions
 * @param objects - Array of scene objects to render and allow dragging for position updates
 * @param onObjectUpdate - Callback invoked with an object's `id` and partial updated properties (positions are reported in meters) when an object is moved
 * @returns The React element for the interactive 2D canvas
 */
export function Canvas2D({ width, height, roomDimensions, objects, onObjectUpdate, selectedObjectId, onSelectObject }: Canvas2DProps) {
    const stageRef = useRef<any>(null);
    const [scale, setScale] = useState(1);
    const [position, setPosition] = useState({ x: STAGE_PADDING, y: STAGE_PADDING });

    // Calculate room size in pixels
    const roomWidthPx = roomDimensions.width * GRID_SIZE;
    const roomLengthPx = roomDimensions.length * GRID_SIZE;

    // Handle Zoom
    const handleWheel = (e: any) => {
        e.evt.preventDefault();
        const scaleBy = 1.1;
        const stage = stageRef.current;
        const oldScale = stage.scaleX();
        const mousePointTo = {
            x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
            y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale,
        };

        const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;

        stage.scale({ x: newScale, y: newScale });

        const newPos = {
            x: -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
            y: -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale,
        };
        setPosition(newPos);
        setScale(newScale);
    };

    return (
        <div className="bg-slate-100 overflow-hidden rounded-lg border shadow-inner">
            <Stage
                width={width}
                height={height}
                onWheel={handleWheel}
                scaleX={scale}
                scaleY={scale}
                x={position.x}
                y={position.y}
                draggable
                ref={stageRef}
            >
                <Layer>
                    {/* Grid */}
                    <Grid width={roomWidthPx} height={roomLengthPx} />

                    {/* Room Boundary */}
                    <Rect
                        x={0}
                        y={0}
                        width={roomWidthPx}
                        height={roomLengthPx}
                        stroke="black"
                        strokeWidth={4}
                        fill="white"
                        opacity={0.5}
                    />

                    {/* Objects */}
                    {objects.map((obj) => (
                        <DraggableObject
                            key={obj.id}
                            obj={obj}
                            selected={selectedObjectId === obj.id}
                            onUpdate={onObjectUpdate}
                            onSelect={onSelectObject}
                        />
                    ))}
                </Layer>
            </Stage>
            <div className="absolute bottom-4 right-4 bg-white/80 p-2 rounded text-xs">
                Scroll to Zoom • Drag to Pan
            </div>
        </div>
    );
}

const Grid = ({ width, height }: { width: number; height: number }) => {
    const lines = [];
    // Vertical lines
    for (let i = 0; i <= width / GRID_SIZE; i++) {
        lines.push(
            <Line
                key={`v-${i}`}
                points={[i * GRID_SIZE, 0, i * GRID_SIZE, height]}
                stroke="#ddd"
                strokeWidth={1}
            />
        );
    }
    // Horizontal lines
    for (let i = 0; i <= height / GRID_SIZE; i++) {
        lines.push(
            <Line
                key={`h-${i}`}
                points={[0, i * GRID_SIZE, width, i * GRID_SIZE]}
                stroke="#ddd"
                strokeWidth={1}
            />
        );
    }
    return <Group>{lines}</Group>;
};

const DraggableObject = ({ obj, onUpdate, selected, onSelect }: { obj: SceneObject2D; onUpdate: (id: string, newProps: Partial<SceneObject2D>) => void; selected?: boolean; onSelect?: (id: string) => void; }) => {
    const shapeRef = useRef<any>(null);
    const trRef = useRef<any>(null);

    // Convert meters to pixels
    const widthPx = obj.dimensions.width * GRID_SIZE;
    const depthPx = obj.dimensions.depth * GRID_SIZE;
    const xPx = obj.position.x * GRID_SIZE;
    const yPx = obj.position.y * GRID_SIZE;

    return (
        <Group
            x={xPx}
            y={yPx}
            rotation={obj.rotation}
            draggable
            onClick={() => onSelect?.(obj.id)}
            onDragEnd={(e) => {
                onUpdate(obj.id, {
                    position: {
                        x: e.target.x() / GRID_SIZE,
                        y: e.target.y() / GRID_SIZE,
                    },
                });
            }}
        >
            <Rect
                width={widthPx}
                height={depthPx}
                fill={obj.color || '#4f46e5'}
                stroke={selected ? '#2563eb' : 'black'}
                strokeWidth={selected ? 3 : 2}
                offsetX={widthPx / 2}
                offsetY={depthPx / 2}
                cornerRadius={4}
            />
            <Text
                text={obj.type}
                fontSize={12}
                fill="white"
                offsetX={widthPx / 2}
                offsetY={depthPx / 2}
                width={widthPx}
                height={depthPx}
                align="center"
                verticalAlign="middle"
            />
        </Group>
    );
};
