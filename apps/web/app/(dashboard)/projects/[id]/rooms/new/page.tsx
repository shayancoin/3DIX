'use client';

import { useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ArrowLeft, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { RoomType as SchemaRoomType } from '@/lib/db/schema';
import { RoomType, getRoomTypeConfig } from '@3dix/types';
import { RoomWizard } from '@/components/wizards/RoomWizard';

/**
 * Convert a schema-defined RoomType enum value to the UI `RoomType` string.
 *
 * @param schemaType - The schema `SchemaRoomType` value to convert
 * @returns The corresponding `RoomType` string; returns `'other'` if the input has no match
 */
function mapSchemaRoomTypeToType(schemaType: SchemaRoomType): RoomType {
  const mapping: Record<string, RoomType> = {
    KITCHEN: 'kitchen',
    BEDROOM: 'bedroom',
    BATHROOM: 'bathroom',
    LIVING_ROOM: 'living_room',
    DINING_ROOM: 'dining_room',
    OFFICE: 'office',
    CLOSET: 'closet',
    OTHER: 'other',
  };
  return mapping[schemaType] || 'other';
}

/**
 * Map a UI `RoomType` value to the corresponding `SchemaRoomType` enum value.
 *
 * @param type - The UI room type to convert (e.g., 'kitchen', 'bedroom', 'living_room').
 * @returns The matching `SchemaRoomType` enum value.
 */
function mapTypeRoomTypeToSchema(type: RoomType): SchemaRoomType {
  const mapping: Record<RoomType, SchemaRoomType> = {
    kitchen: SchemaRoomType.KITCHEN,
    bedroom: SchemaRoomType.BEDROOM,
    bathroom: SchemaRoomType.BATHROOM,
    living_room: SchemaRoomType.LIVING_ROOM,
    dining_room: SchemaRoomType.DINING_ROOM,
    office: SchemaRoomType.OFFICE,
    closet: SchemaRoomType.CLOSET,
    other: SchemaRoomType.OTHER,
  };
  return mapping[type];
}

/**
 * Render the "Create New Room" page with a form and an optional wizard workflow.
 *
 * The component manages local form state, toggles between a traditional form and a wizard, submits room creation requests to the project rooms API, and navigates to the created room on success. It also exposes client-side loading and error states.
 *
 * @returns The React element for the New Room page UI containing the form, wizard entry, and related controls.
 */
export default function NewRoomPage() {
  const router = useRouter();
  const params = useParams();
  const projectId = params.projectId as string;

  const [useWizard, setUseWizard] = useState(false);
  const [wizardRoomType, setWizardRoomType] = useState<RoomType>('kitchen');
  const [name, setName] = useState('');
  const [roomType, setRoomType] = useState<SchemaRoomType>(SchemaRoomType.KITCHEN);
  const [width, setWidth] = useState('');
  const [height, setHeight] = useState('');
  const [length, setLength] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/projects/${projectId}/rooms`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name,
          roomType,
          width: width ? parseFloat(width) : undefined,
          height: height ? parseFloat(height) : undefined,
          length: length ? parseFloat(length) : undefined,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to create room');
      }

      const room = await response.json();
      router.push(`/projects/${projectId}/rooms/${room.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleWizardComplete = async (data: {
    name: string;
    dimensions: { width: number; length: number; height: number };
    selectedCategories: string[];
  }) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/projects/${projectId}/rooms`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: data.name,
          roomType: mapTypeRoomTypeToSchema(wizardRoomType),
          width: data.dimensions.width,
          height: data.dimensions.height,
          length: data.dimensions.length,
          selectedCategories: data.selectedCategories, // Store in metadata
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to create room');
      }

      const room = await response.json();
      router.push(`/projects/${projectId}/rooms/${room.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setLoading(false);
    }
  };

  if (useWizard) {
    return (
      <div className="container mx-auto p-6">
        <Link href={`/projects/${projectId}`} className="inline-flex items-center text-muted-foreground hover:text-foreground mb-6">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Project
        </Link>
        <RoomWizard
          roomType={wizardRoomType}
          onComplete={handleWizardComplete}
          onCancel={() => setUseWizard(false)}
        />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-2xl">
      <Link href={`/projects/${projectId}`} className="inline-flex items-center text-muted-foreground hover:text-foreground mb-6">
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Project
      </Link>

      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold">Create New Room</h1>
          <Button
            variant="outline"
            onClick={() => {
              setWizardRoomType(mapSchemaRoomTypeToType(roomType));
              setUseWizard(true);
            }}
            className="flex items-center gap-2"
          >
            <Sparkles className="h-4 w-4" />
            Use Wizard
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="name">Room Name *</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              placeholder="Main Kitchen"
            />
          </div>

          <div>
            <Label htmlFor="roomType">Room Type *</Label>
            <select
              id="roomType"
              value={roomType}
              onChange={(e) => setRoomType(e.target.value as RoomType)}
              className="w-full px-3 py-2 border border-input bg-background rounded-md"
              required
            >
              {Object.values(RoomType).map((type) => (
                <option key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label htmlFor="width">Width (m)</Label>
              <Input
                id="width"
                type="number"
                step="0.1"
                value={width}
                onChange={(e) => setWidth(e.target.value)}
                placeholder="5.0"
              />
            </div>
            <div>
              <Label htmlFor="height">Height (m)</Label>
              <Input
                id="height"
                type="number"
                step="0.1"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
                placeholder="2.5"
              />
            </div>
            <div>
              <Label htmlFor="length">Length (m)</Label>
              <Input
                id="length"
                type="number"
                step="0.1"
                value={length}
                onChange={(e) => setLength(e.target.value)}
                placeholder="4.0"
              />
            </div>
          </div>

          {error && (
            <div className="text-destructive text-sm">{error}</div>
          )}

          <div className="flex gap-4">
            <Button type="submit" disabled={loading}>
              {loading ? 'Creating...' : 'Create Room'}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => router.back()}
            >
              Cancel
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
}