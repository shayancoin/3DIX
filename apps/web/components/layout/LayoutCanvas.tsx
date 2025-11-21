'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { Canvas2D } from './Canvas2D';
import { SceneObject2D, CanvasViewport, LayoutCanvasState } from '@3dix/types';
import { useSceneHistory } from '@/hooks/useSceneHistory';

interface LayoutCanvasProps {
  initialObjects?: SceneObject2D[];
  initialViewport?: CanvasViewport;
  roomWidth?: number;
  roomLength?: number;
  onStateChange?: (state: LayoutCanvasState) => void;
}

export function LayoutCanvas({
  initialObjects = [],
  initialViewport = { x: 0, y: 0, zoom: 1 },
  roomWidth = 5,
  roomLength = 4,
  onStateChange,
}: LayoutCanvasProps) {
  const [objects, setObjects] = useState<SceneObject2D[]>(initialObjects);
  const [viewport, setViewport] = useState<CanvasViewport>(initialViewport);
  const [selectedObjectId, setSelectedObjectId] = useState<string | undefined>();
  const { history, historyIndex, addHistoryEntry, undo, redo, canUndo, canRedo } = useSceneHistory();

  // Update objects when initialObjects change
  useEffect(() => {
    setObjects(initialObjects);
  }, [initialObjects]);

  // Notify parent of state changes
  useEffect(() => {
    onStateChange?.({
      objects,
      viewport,
      selectedObjectId,
      history,
      historyIndex,
    });
  }, [objects, viewport, selectedObjectId, history, historyIndex, onStateChange]);

  const handleObjectsChange = useCallback(
    (newObjects: SceneObject2D[]) => {
      const previousObjects = objects;
      setObjects(newObjects);

      // Create history entry for object changes
      if (previousObjects.length !== newObjects.length) {
        // Object added or deleted
        const added = newObjects.find((o) => !previousObjects.find((p) => p.id === o.id));
        const deleted = previousObjects.find((o) => !newObjects.find((p) => p.id === o.id));

        if (added) {
          addHistoryEntry({
            action: 'create',
            objectId: added.id,
            newState: added,
          });
        } else if (deleted) {
          addHistoryEntry({
            action: 'delete',
            objectId: deleted.id,
            previousState: deleted,
          });
        }
      } else {
        // Object moved or updated
        const changed = newObjects.find(
          (o, i) =>
            o.position.x !== previousObjects[i]?.position.x ||
            o.position.y !== previousObjects[i]?.position.y
        );
        if (changed) {
          const previous = previousObjects.find((o) => o.id === changed.id);
          addHistoryEntry({
            action: 'move',
            objectId: changed.id,
            previousState: previous,
            newState: changed,
          });
        }
      }
    },
    [objects, addHistoryEntry]
  );

  const handleViewportChange = useCallback((newViewport: CanvasViewport) => {
    setViewport(newViewport);
  }, []);

  const handleObjectSelect = useCallback((objectId: string | undefined) => {
    setSelectedObjectId(objectId);
  }, []);

  const handleAddObject = useCallback(
    (category: string, position: { x: number; y: number }, size: { width: number; height: number }) => {
      const newObject: SceneObject2D = {
        id: `obj-${Date.now()}`,
        category,
        position,
        size,
        boundingBox: {
          x: position.x,
          y: position.y,
          width: size.width,
          height: size.height,
        },
        label: category,
      };

      setObjects([...objects, newObject]);
      addHistoryEntry({
        action: 'create',
        objectId: newObject.id,
        newState: newObject,
      });
    },
    [objects, addHistoryEntry]
  );

  const handleDeleteSelected = useCallback(() => {
    if (selectedObjectId) {
      const objectToDelete = objects.find((o) => o.id === selectedObjectId);
      if (objectToDelete) {
        setObjects(objects.filter((o) => o.id !== selectedObjectId));
        addHistoryEntry({
          action: 'delete',
          objectId: selectedObjectId,
          previousState: objectToDelete,
        });
        setSelectedObjectId(undefined);
      }
    }
  }, [selectedObjectId, objects, addHistoryEntry]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 relative min-h-0">
        <Canvas2D
          width={800}
          height={600}
          objects={objects}
          viewport={viewport}
          onObjectsChange={handleObjectsChange}
          onViewportChange={handleViewportChange}
          selectedObjectId={selectedObjectId}
          onObjectSelect={handleObjectSelect}
          roomWidth={roomWidth}
          roomLength={roomLength}
        />
      </div>
      <div className="flex items-center gap-2 p-2 border-t bg-white">
        <button
          onClick={undo}
          disabled={!canUndo}
          className="px-3 py-1 text-sm border rounded disabled:opacity-50"
        >
          Undo
        </button>
        <button
          onClick={redo}
          disabled={!canRedo}
          className="px-3 py-1 text-sm border rounded disabled:opacity-50"
        >
          Redo
        </button>
        <div className="flex-1" />
        <button
          onClick={handleDeleteSelected}
          disabled={!selectedObjectId}
          className="px-3 py-1 text-sm border rounded text-red-600 disabled:opacity-50"
        >
          Delete Selected
        </button>
      </div>
    </div>
  );
}

// Export for use in other components
export { LayoutCanvas };
