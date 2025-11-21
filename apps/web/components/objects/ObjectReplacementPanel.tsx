'use client';

import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, X, Loader2 } from 'lucide-react';
import { Card } from '@/components/ui/card';

interface ObjectReplacementPanelProps {
  selectedObjectId: string | undefined;
  onObjectReplaced?: (objectId: string, meshData: string) => void;
  onClose?: () => void;
}

export function ObjectReplacementPanel({
  selectedObjectId,
  onObjectReplaced,
  onClose,
}: ObjectReplacementPanelProps) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [maskFile, setMaskFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [maskPreview, setMaskPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconstructedMesh, setReconstructedMesh] = useState<string | null>(null);

  const handleImageChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (event) => {
      setImagePreview(event.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleMaskChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setMaskFile(file);
    const reader = new FileReader();
    reader.onload = (event) => {
      setMaskPreview(event.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleReconstruct = useCallback(async () => {
    if (!imageFile || !selectedObjectId) {
      setError('Please select an object and upload an image');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Convert image to base64
      const imageBase64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(imageFile);
      });

      // Convert mask to base64 if provided
      let maskBase64: string | undefined;
      if (maskFile) {
        maskBase64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(maskFile);
        });
      }

      // Call reconstruction API
      const response = await fetch('/api/custom-objects', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
          mask: maskBase64,
          mask_type: maskBase64 ? 'single' : undefined,
          seed: 42,
          output_format: 'gltf',
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to reconstruct object');
      }

      const data = await response.json();
      setReconstructedMesh(data.mesh_data || data.mesh_url);

      // Notify parent component
      if (data.mesh_data || data.mesh_url) {
        onObjectReplaced?.(selectedObjectId, data.mesh_data || data.mesh_url);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [imageFile, maskFile, selectedObjectId, onObjectReplaced]);

  if (!selectedObjectId) {
    return (
      <Card className="p-4">
        <p className="text-sm text-muted-foreground text-center">
          Select an object to replace it with a custom 3D model
        </p>
      </Card>
    );
  }

  return (
    <Card className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Replace Object</h3>
        {onClose && (
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <div>
        <Label htmlFor="object-image">Object Image *</Label>
        <Input
          id="object-image"
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="mt-2"
        />
        {imagePreview && (
          <div className="mt-2">
            <img
              src={imagePreview}
              alt="Object preview"
              className="max-w-full h-32 object-contain border rounded"
            />
          </div>
        )}
      </div>

      <div>
        <Label htmlFor="object-mask">Mask Image (optional)</Label>
        <Input
          id="object-mask"
          type="file"
          accept="image/*"
          onChange={handleMaskChange}
          className="mt-2"
        />
        {maskPreview && (
          <div className="mt-2">
            <img
              src={maskPreview}
              alt="Mask preview"
              className="max-w-full h-32 object-contain border rounded"
            />
            <button
              onClick={() => {
                setMaskFile(null);
                setMaskPreview(null);
              }}
              className="mt-1 text-xs text-red-600 hover:text-red-800"
            >
              Remove mask
            </button>
          </div>
        )}
        <p className="text-xs text-muted-foreground mt-1">
          Upload a mask to specify which part of the image to reconstruct
        </p>
      </div>

      {error && (
        <div className="text-sm text-red-600 bg-red-50 p-2 rounded">
          {error}
        </div>
      )}

      {reconstructedMesh && (
        <div className="text-sm text-green-600 bg-green-50 p-2 rounded">
          Object reconstructed successfully! The mesh has been applied to the selected object.
        </div>
      )}

      <Button
        onClick={handleReconstruct}
        disabled={loading || !imageFile}
        className="w-full"
      >
        {loading ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Reconstructing...
          </>
        ) : (
          <>
            <Upload className="h-4 w-4 mr-2" />
            Reconstruct & Replace
          </>
        )}
      </Button>

      <p className="text-xs text-muted-foreground">
        This will use SAM-3D to reconstruct a 3D model from your image and replace the selected object.
      </p>
    </Card>
  );
}
