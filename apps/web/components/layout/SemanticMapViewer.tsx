'use client';

import React, { useEffect, useRef, useState } from 'react';
import { LayoutObject } from '@3dix/types';

interface SemanticMapViewerProps {
  semanticMapUrl?: string; // base64 data URL
  objects: LayoutObject[];
  roomWidth?: number;
  roomLength?: number;
  onObjectClick?: (objectId: string) => void;
  selectedObjectId?: string;
  selectedId?: string;
}

export function SemanticMapViewer({
  semanticMapUrl,
  objects,
  roomWidth = 5,
  roomLength = 4,
  onObjectClick,
  selectedObjectId,
  selectedId,
}: SemanticMapViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const width = 800;
    const height = 600;
    canvas.width = width;
    canvas.height = height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw semantic map if available
    if (semanticMapUrl) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, width, height);
        drawBoundingBoxes(ctx, width, height);
        setImageLoaded(true);
      };
      img.onerror = () => {
        // If image fails to load, just draw bounding boxes
        drawBoundingBoxes(ctx, width, height);
        setImageLoaded(true);
      };
      img.src = semanticMapUrl;
    } else {
      // Draw room boundary and bounding boxes
      drawRoomBoundary(ctx, width, height);
      drawBoundingBoxes(ctx, width, height);
      setImageLoaded(true);
    }

    function drawRoomBoundary(ctx: CanvasRenderingContext2D, w: number, h: number) {
      const scale = Math.min(w / roomWidth, h / roomLength) * 0.8;
      const offsetX = (w - roomWidth * scale) / 2;
      const offsetY = (h - roomLength * scale) / 2;

      ctx.strokeStyle = '#CCCCCC';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(offsetX, offsetY, roomWidth * scale, roomLength * scale);
      ctx.setLineDash([]);
    }

    function drawBoundingBoxes(ctx: CanvasRenderingContext2D, w: number, h: number) {
      const scale = Math.min(w / roomWidth, h / roomLength) * 0.8;
      const offsetX = (w - roomWidth * scale) / 2;
      const offsetY = (h - roomLength * scale) / 2;

      objects.forEach((obj) => {
        const isSelected = obj.id === (selectedObjectId ?? selectedId);
        const x = offsetX + obj.position[0] * scale;
        const z = offsetY + obj.position[2] * scale;
        const objWidth = obj.size[0] * scale;
        const objDepth = obj.size[2] * scale;

        // Draw bounding box
        ctx.strokeStyle = isSelected ? '#3B82F6' : '#000000';
        ctx.lineWidth = isSelected ? 3 : 1;
        ctx.fillStyle = isSelected ? 'rgba(59, 130, 246, 0.2)' : 'rgba(0, 0, 0, 0.1)';
        
        // Apply rotation
        ctx.save();
        ctx.translate(x + objWidth / 2, z + objDepth / 2);
        const angle = Math.abs(obj.orientation) <= 3 ? obj.orientation * (Math.PI / 2) : obj.orientation;
        ctx.rotate(angle);
        ctx.fillRect(-objWidth / 2, -objDepth / 2, objWidth, objDepth);
        ctx.strokeRect(-objWidth / 2, -objDepth / 2, objWidth, objDepth);
        ctx.restore();

        // Draw label
        ctx.fillStyle = '#000000';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(obj.category, x + objWidth / 2, z - 5);
      });
    }
  }, [semanticMapUrl, objects, roomWidth, roomLength, selectedObjectId]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onObjectClick) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const scale = Math.min(canvas.width / roomWidth, canvas.height / roomLength) * 0.8;
    const offsetX = (canvas.width - roomWidth * scale) / 2;
    const offsetY = (canvas.height - roomLength * scale) / 2;

    // Find clicked object
    const clickedObj = objects.find((obj) => {
      const objX = offsetX + obj.position[0] * scale;
      const objZ = offsetY + obj.position[2] * scale;
      const objWidth = obj.size[0] * scale;
      const objDepth = obj.size[2] * scale;

      return (
        x >= objX - objWidth / 2 &&
        x <= objX + objWidth / 2 &&
        y >= objZ - objDepth / 2 &&
        y <= objZ + objDepth / 2
      );
    });

    if (clickedObj) {
      onObjectClick(clickedObj.id);
    } else {
      onObjectClick(undefined);
    }
  };

  return (
    <div className="relative w-full h-full bg-gray-100 border border-gray-300 rounded-lg overflow-hidden">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="w-full h-full cursor-pointer"
        style={{ imageRendering: 'pixelated' }}
      />
      {!imageLoaded && semanticMapUrl && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
          <div className="text-muted-foreground">Loading semantic map...</div>
        </div>
      )}
    </div>
  );
}
