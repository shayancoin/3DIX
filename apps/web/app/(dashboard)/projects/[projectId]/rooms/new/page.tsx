'use client';

import { useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import { RoomType } from '@/lib/db/schema';

export default function NewRoomPage() {
  const router = useRouter();
  const params = useParams();
  const projectId = params.projectId as string;

  const [name, setName] = useState('');
  const [roomType, setRoomType] = useState<RoomType>(RoomType.KITCHEN);
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

  return (
    <div className="container mx-auto p-6 max-w-2xl">
      <Link href={`/projects/${projectId}`} className="inline-flex items-center text-muted-foreground hover:text-foreground mb-6">
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Project
      </Link>

      <Card className="p-6">
        <h1 className="text-3xl font-bold mb-6">Create New Room</h1>

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
