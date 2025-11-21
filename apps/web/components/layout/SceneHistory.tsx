'use client';

import React, { useState, useEffect } from 'react';
import { RoomGeneration } from '@/lib/db/schema';
import { VibeSpec } from '@3dix/types';
import { Clock, Eye, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface SceneHistoryProps {
  roomId: number;
  onGenerationSelect?: (generation: RoomGeneration) => void;
  onGenerationDelete?: (generationId: number) => void;
}

export function SceneHistory({ roomId, onGenerationSelect, onGenerationDelete }: SceneHistoryProps) {
  const [generations, setGenerations] = useState<RoomGeneration[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  useEffect(() => {
    loadGenerations();
  }, [roomId]);

  const loadGenerations = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/rooms/${roomId}/generations`);
      if (response.ok) {
        const data = await response.json();
        setGenerations(data);
      }
    } catch (error) {
      console.error('Error loading generations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = (generation: RoomGeneration) => {
    setSelectedId(generation.id);
    onGenerationSelect?.(generation);
  };

  const handleDelete = async (generationId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const response = await fetch(`/api/rooms/${roomId}/generations/${generationId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setGenerations(generations.filter((g) => g.id !== generationId));
        onGenerationDelete?.(generationId);
        if (selectedId === generationId) {
          setSelectedId(null);
        }
      }
    } catch (error) {
      console.error('Error deleting generation:', error);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getVibePreview = (vibeSpec: VibeSpec | null) => {
    if (!vibeSpec) return 'No vibe specified';
    const text = vibeSpec.prompt?.text || '';
    if (text.length > 50) {
      return text.substring(0, 50) + '...';
    }
    return text || 'Empty vibe';
  };

  if (loading) {
    return (
      <div className="p-4 text-sm text-muted-foreground">Loading history...</div>
    );
  }

  if (generations.length === 0) {
    return (
      <div className="p-4 text-sm text-muted-foreground text-center">
        <p>No generations yet</p>
        <p className="text-xs mt-1">Create a vibe and generate a layout to see history</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b bg-white">
        <h3 className="text-lg font-semibold">Scene History</h3>
        <p className="text-xs text-muted-foreground mt-1">
          {generations.length} {generations.length === 1 ? 'generation' : 'generations'}
        </p>
      </div>
      <div className="flex-1 overflow-y-auto">
        <div className="divide-y">
          {generations.map((generation) => {
            const vibeSpec = generation.vibeSpec as VibeSpec | null;
            const isSelected = selectedId === generation.id;
            
            return (
              <div
                key={generation.id}
                onClick={() => handleSelect(generation)}
                className={`p-4 cursor-pointer transition-colors hover:bg-gray-50 ${
                  isSelected ? 'bg-primary/5 border-l-4 border-primary' : ''
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <Clock className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                      <span className="text-xs text-muted-foreground">
                        {formatDate(generation.createdAt.toString())}
                      </span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded ${
                          generation.status === 'completed'
                            ? 'bg-green-100 text-green-700'
                            : generation.status === 'generated'
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-gray-100 text-gray-700'
                        }`}
                      >
                        {generation.status}
                      </span>
                    </div>
                    <p className="text-sm text-foreground line-clamp-2">
                      {getVibePreview(vibeSpec)}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      onClick={(e) => handleDelete(generation.id, e)}
                    >
                      <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                    </Button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
