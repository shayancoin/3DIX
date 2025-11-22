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
  selectedObjectId?: string;
  onObjectSelect?: (objectId: string | undefined) => void;
}

export function LayoutCanvas({
  initialObjects = [],
  initialViewport = { x: 0, y: 0, zoom: 1 },
  roomWidth = 5,
  roomLength = 4,
  onStateChange,
  selectedObjectId,
  onObjectSelect,
}: LayoutCanvasProps) {
  const [objects, setObjects] = useState<SceneObject2D[]>(initialObjects);
  const [viewport, setViewport] = useState<CanvasViewport>(initialViewport);
  const [internalSelectedId, setInternalSelectedId] = useState<string | undefined>(selectedObjectId);
  const { history, historyIndex, addHistoryEntry, undo, redo, canUndo, canRedo } = useSceneHistory();

  // Sync external selection
  useEffect(() => {
    setInternalSelectedId(selectedObjectId);
  }, [selectedObjectId]);

  // Update objects when initialObjects change
  useEffect(() => {
    setObjects(initialObjects);
  }, [initialObjects]);


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

  // Update state when notifying parent
  useEffect(() => {
    onStateChange?.({
      objects,
      viewport,
      selectedObjectId: internalSelectedId,
      history,
      historyIndex,
    });
  }, [objects, viewport, internalSelectedId, history, historyIndex, onStateChange]);

  const handleViewportChange = useCallback((newViewport: CanvasViewport) => {
    setViewport(newViewport);
  }, []);

  const handleObjectSelect = useCallback((objectId: string | undefined) => {
    const newSelectedId = objectId === internalSelectedId ? undefined : objectId;
    setInternalSelectedId(newSelectedId);
    onObjectSelect?.(newSelectedId);
  }, [internalSelectedId, onObjectSelect]);

  const handleAddObject = useCallback(
    (category: string, position: { x: number; y: number }, dimensions: { width: number; depth: number }) => {
      const newObject: SceneObject2D = {
        id: `obj-${Date.now()}`,
        type: 'object',
        category,
        position,
        rotation: 0,
        dimensions,
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
    if (internalSelectedId) {
      const objectToDelete = objects.find((o) => o.id === internalSelectedId);
      if (objectToDelete) {
        setObjects(objects.filter((o) => o.id !== internalSelectedId));
        addHistoryEntry({
          action: 'delete',
          objectId: internalSelectedId,
          previousState: objectToDelete,
        });
        setInternalSelectedId(undefined);
        onObjectSelect?.(undefined);
      }
    }
  }, [internalSelectedId, objects, addHistoryEntry, onObjectSelect]);

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
          selectedObjectId={internalSelectedId}
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


