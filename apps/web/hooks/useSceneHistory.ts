'use client';

import { useState, useCallback } from 'react';
import { SceneHistoryEntry, SceneObject2D, CanvasViewport } from '@3dix/types';

const MAX_HISTORY = 50;

export function useSceneHistory() {
  const [history, setHistory] = useState<SceneHistoryEntry[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const addHistoryEntry = useCallback(
    (entry: Omit<SceneHistoryEntry, 'id' | 'timestamp'>) => {
      const newEntry: SceneHistoryEntry = {
        ...entry,
        id: `hist-${Date.now()}-${Math.random()}`,
        timestamp: new Date().toISOString(),
      };

      // Remove any history after current index (when user makes new change after undo)
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push(newEntry);

      // Limit history size
      if (newHistory.length > MAX_HISTORY) {
        newHistory.shift();
      } else {
        setHistoryIndex(newHistory.length - 1);
      }

      setHistory(newHistory);
    },
    [history, historyIndex]
  );

  const undo = useCallback(() => {
    if (historyIndex >= 0) {
      setHistoryIndex(historyIndex - 1);
    }
  }, [historyIndex]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
    }
  }, [historyIndex, history.length]);

  const canUndo = historyIndex >= 0;
  const canRedo = historyIndex < history.length - 1;

  const clearHistory = useCallback(() => {
    setHistory([]);
    setHistoryIndex(-1);
  }, []);

  return {
    history,
    historyIndex,
    addHistoryEntry,
    undo,
    redo,
    canUndo,
    canRedo,
    clearHistory,
  };
}
