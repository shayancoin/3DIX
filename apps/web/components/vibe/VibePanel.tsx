'use client';

import React, { useState, useCallback } from 'react';
import { VibeSpec, VibeTag, VibeSlider, RoomType } from '@3dix/types';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { X, Plus, Tag } from 'lucide-react';

interface VibePanelProps {
  initialVibeSpec?: VibeSpec;
  roomType: RoomType;
  roomId?: number;
  onVibeSpecChange?: (vibeSpec: VibeSpec) => void;
  onSubmit?: (vibeSpec: VibeSpec) => void;
  onJobCreated?: (jobId: number) => void;
}

// Predefined tag options
const TAG_OPTIONS: Record<string, VibeTag[]> = {
  style: [
    { id: 'modern', label: 'Modern', category: 'style', weight: 0.5 },
    { id: 'minimalist', label: 'Minimalist', category: 'style', weight: 0.5 },
    { id: 'rustic', label: 'Rustic', category: 'style', weight: 0.5 },
    { id: 'industrial', label: 'Industrial', category: 'style', weight: 0.5 },
    { id: 'scandinavian', label: 'Scandinavian', category: 'style', weight: 0.5 },
    { id: 'traditional', label: 'Traditional', category: 'style', weight: 0.5 },
  ],
  mood: [
    { id: 'cozy', label: 'Cozy', category: 'mood', weight: 0.5 },
    { id: 'bright', label: 'Bright', category: 'mood', weight: 0.5 },
    { id: 'calm', label: 'Calm', category: 'mood', weight: 0.5 },
    { id: 'energetic', label: 'Energetic', category: 'mood', weight: 0.5 },
    { id: 'luxurious', label: 'Luxurious', category: 'mood', weight: 0.5 },
  ],
  color: [
    { id: 'warm', label: 'Warm Tones', category: 'color', weight: 0.5 },
    { id: 'cool', label: 'Cool Tones', category: 'color', weight: 0.5 },
    { id: 'neutral', label: 'Neutral', category: 'color', weight: 0.5 },
    { id: 'bold', label: 'Bold Colors', category: 'color', weight: 0.5 },
  ],
  material: [
    { id: 'wood', label: 'Wood', category: 'material', weight: 0.5 },
    { id: 'metal', label: 'Metal', category: 'material', weight: 0.5 },
    { id: 'stone', label: 'Stone', category: 'material', weight: 0.5 },
    { id: 'glass', label: 'Glass', category: 'material', weight: 0.5 },
  ],
};

const DEFAULT_SLIDERS: VibeSlider[] = [
  { id: 'spaciousness', label: 'Spaciousness', min: 0, max: 1, value: 0.5, step: 0.1 },
  { id: 'complexity', label: 'Complexity', min: 0, max: 1, value: 0.5, step: 0.1 },
  { id: 'formality', label: 'Formality', min: 0, max: 1, value: 0.5, step: 0.1 },
];

export function VibePanel({
  initialVibeSpec,
  roomType,
  roomId,
  onVibeSpecChange,
  onSubmit,
  onJobCreated,
}: VibePanelProps) {
  const [prompt, setPrompt] = useState(initialVibeSpec?.prompt.text || '');
  const [referenceImageUrl, setReferenceImageUrl] = useState(initialVibeSpec?.prompt.referenceImageUrl || '');
  const [tags, setTags] = useState<VibeTag[]>(initialVibeSpec?.tags || []);
  const [sliders, setSliders] = useState<VibeSlider[]>(initialVibeSpec?.sliders || DEFAULT_SLIDERS);
  const [maskType, setMaskType] = useState<'none' | 'room_boundary' | 'wall_mask' | 'door_window_mask'>('none');
  const [maskImagePreview, setMaskImagePreview] = useState<string | null>(null);
  const [maskImageBase64, setMaskImageBase64] = useState<string | null>(null);

  const handlePromptChange = useCallback(
    (text: string) => {
      setPrompt(text);
      updateVibeSpec({ prompt: { text, referenceImageUrl, roomType } });
    },
    [referenceImageUrl, roomType]
  );

  const handleReferenceImageChange = useCallback(
    (url: string) => {
      setReferenceImageUrl(url);
      updateVibeSpec({ prompt: { text: prompt, referenceImageUrl: url, roomType } });
    },
    [prompt, roomType]
  );

  const handleMaskImageChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result;
        if (typeof result === 'string') {
          setMaskImagePreview(result);
          setMaskImageBase64(result);
        }
      };
      reader.readAsDataURL(file);
    },
    []
  );

  const handleTagToggle = useCallback(
    (tag: VibeTag) => {
      const isSelected = tags.some((t) => t.id === tag.id);
      const newTags = isSelected
        ? tags.filter((t) => t.id !== tag.id)
        : [...tags, tag];
      setTags(newTags);
      updateVibeSpec({ tags: newTags });
    },
    [tags]
  );

  const handleTagRemove = useCallback(
    (tagId: string) => {
      const newTags = tags.filter((t) => t.id !== tagId);
      setTags(newTags);
      updateVibeSpec({ tags: newTags });
    },
    [tags]
  );

  const handleSliderChange = useCallback(
    (sliderId: string, value: number) => {
      const newSliders = sliders.map((s) => (s.id === sliderId ? { ...s, value } : s));
      setSliders(newSliders);
      updateVibeSpec({ sliders: newSliders });
    },
    [sliders]
  );

  const updateVibeSpec = useCallback(
    (updates: Partial<VibeSpec>) => {
      const newVibeSpec: VibeSpec = {
        prompt: { text: prompt, referenceImageUrl, roomType },
        tags,
        sliders,
        ...updates,
        metadata: {
          updatedAt: new Date().toISOString(),
        },
      };
      onVibeSpecChange?.(newVibeSpec);
    },
    [prompt, referenceImageUrl, roomType, tags, sliders, onVibeSpecChange]
  );

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleSubmit = useCallback(async () => {
    if (!roomId) {
      setSubmitError('Room ID is required');
      return;
    }

    const vibeSpec: VibeSpec = {
      prompt: { text: prompt, referenceImageUrl, roomType },
      tags,
      sliders,
      metadata: {
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      },
    };

    setSubmitting(true);
    setSubmitError(null);

    try {
      // Prepare constraints with mask information
      const constraints: any = {};
      if (maskType !== 'none') {
        constraints.maskType = maskType;
        if (maskImageBase64) {
          constraints.maskImage = maskImageBase64;
        }
      }

      // Create layout generation job
      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          roomId,
          requestData: {
            roomId: roomId.toString(),
            vibeSpec,
            constraints: Object.keys(constraints).length > 0 ? constraints : undefined,
          },
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to create job');
      }

      const job = await response.json();
      onJobCreated?.(job.id);
      onSubmit?.(vibeSpec);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setSubmitting(false);
    }
  }, [prompt, referenceImageUrl, roomType, tags, sliders, roomId, onSubmit, onJobCreated]);

  return (
    <div className="flex flex-col h-full bg-white border-l border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-semibold mb-2">Vibe Specification</h2>
        <p className="text-sm text-muted-foreground">Define the style and mood for your room</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Prompt Section */}
        <div>
          <Label htmlFor="prompt">Description Prompt</Label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => handlePromptChange(e.target.value)}
            placeholder="Describe the desired style, mood, and atmosphere..."
            className="w-full min-h-[100px] px-3 py-2 border border-input bg-background rounded-md mt-2"
          />
        </div>

        {/* Reference Image Section */}
        <div>
          <Label htmlFor="referenceImage">Reference Image URL (optional)</Label>
          <Input
            id="referenceImage"
            type="url"
            value={referenceImageUrl}
            onChange={(e) => handleReferenceImageChange(e.target.value)}
            placeholder="https://example.com/image.jpg"
            className="mt-2"
          />
        </div>

        {/* Mask Type Section */}
        <div>
          <Label>Conditioning Mask Type</Label>
          <div className="mt-2 space-y-2">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="maskType"
                value="none"
                checked={maskType === 'none'}
                onChange={(e) => setMaskType(e.target.value as any)}
                className="w-4 h-4"
              />
              <span className="text-sm">None (No conditioning)</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="maskType"
                value="room_boundary"
                checked={maskType === 'room_boundary'}
                onChange={(e) => setMaskType(e.target.value as any)}
                className="w-4 h-4"
              />
              <span className="text-sm">Room Boundary</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="maskType"
                value="wall_mask"
                checked={maskType === 'wall_mask'}
                onChange={(e) => setMaskType(e.target.value as any)}
                className="w-4 h-4"
              />
              <span className="text-sm">Wall Mask</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="maskType"
                value="door_window_mask"
                checked={maskType === 'door_window_mask'}
                onChange={(e) => setMaskType(e.target.value as any)}
                className="w-4 h-4"
              />
              <span className="text-sm">Door/Window Mask</span>
            </label>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Select the type of conditioning mask to use for layout generation
          </p>
        </div>

        {/* Mask Image Upload */}
        {(maskType === 'room_boundary' || maskType === 'wall_mask' || maskType === 'door_window_mask') && (
          <div>
            <Label htmlFor="maskImage">Upload Mask Image (optional)</Label>
            <input
              id="maskImage"
              type="file"
              accept="image/*"
              onChange={handleMaskImageChange}
              className="mt-2 w-full text-sm"
            />
            {maskImagePreview && (
              <div className="mt-2">
                <img
                  src={maskImagePreview}
                  alt="Mask preview"
                  className="max-w-full h-32 object-contain border rounded"
                />
                <button
                  onClick={() => {
                    setMaskImagePreview(null);
                    setMaskImageBase64(null);
                  }}
                  className="mt-1 text-xs text-red-600 hover:text-red-800"
                >
                  Remove
                </button>
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Upload a mask image to condition the layout generation
            </p>
          </div>
        )}

        {/* Tags Section */}
        <div>
          <Label>Style Tags</Label>
          <div className="mt-2 space-y-3">
            {Object.entries(TAG_OPTIONS).map(([category, categoryTags]) => (
              <div key={category}>
                <div className="text-xs font-medium text-muted-foreground mb-1 capitalize">
                  {category}
                </div>
                <div className="flex flex-wrap gap-2">
                  {categoryTags.map((tag) => {
                    const isSelected = tags.some((t) => t.id === tag.id);
                    return (
                      <button
                        key={tag.id}
                        onClick={() => handleTagToggle(tag)}
                        className={`px-3 py-1 text-sm rounded-full border transition-colors ${
                          isSelected
                            ? 'bg-primary text-primary-foreground border-primary'
                            : 'bg-background border-input hover:bg-accent'
                        }`}
                      >
                        {tag.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {/* Selected Tags */}
          {tags.length > 0 && (
            <div className="mt-3">
              <div className="text-xs font-medium text-muted-foreground mb-2">Selected Tags</div>
              <div className="flex flex-wrap gap-2">
                {tags.map((tag) => (
                  <div
                    key={tag.id}
                    className="px-2 py-1 text-xs bg-primary/10 text-primary rounded flex items-center gap-1"
                  >
                    <Tag className="h-3 w-3" />
                    {tag.label}
                    <button
                      onClick={() => handleTagRemove(tag.id)}
                      className="ml-1 hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Sliders Section */}
        <div>
          <Label>Style Parameters</Label>
          <div className="mt-2 space-y-4">
            {sliders.map((slider) => (
              <div key={slider.id}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium">{slider.label}</span>
                  <span className="text-xs text-muted-foreground">{slider.value.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  value={slider.value}
                  step={slider.step || 0.01}
                  onChange={(e) => handleSliderChange(slider.id, parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <div className="p-4 border-t border-gray-200 space-y-2">
        {submitError && (
          <div className="text-sm text-red-600 bg-red-50 p-2 rounded">
            {submitError}
          </div>
        )}
        <Button
          onClick={handleSubmit}
          className="w-full"
          size="lg"
          disabled={submitting || !roomId}
        >
          {submitting ? 'Creating Job...' : 'Generate Layout'}
        </Button>
        {!roomId && (
          <p className="text-xs text-muted-foreground text-center">
            Room ID is required to generate layout
          </p>
        )}
      </div>
    </div>
  );
}
