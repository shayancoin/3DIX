'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ArrowLeft, Edit, Trash2, DoorOpen } from 'lucide-react';
import Link from 'next/link';

interface Room {
  id: number;
  name: string;
  roomType: string;
  width: number | null;
  height: number | null;
  length: number | null;
  layoutData: any;
  sceneData: any;
  vibeSpec: any;
  thumbnailUrl: string | null;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
  project: {
    id: number;
    name: string;
  };
}

export default function RoomDetailPage() {
  const router = useRouter();
  const params = useParams();
  const projectId = params.projectId as string;
  const roomId = params.id as string;

  const [room, setRoom] = useState<Room | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (projectId && roomId) {
      fetchRoom();
    }
  }, [projectId, roomId]);

  const fetchRoom = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/projects/${projectId}/rooms/${roomId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch room');
      }
      const data = await response.json();
      setRoom(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteRoom = async () => {
    if (!confirm('Are you sure you want to delete this room?')) {
      return;
    }

    try {
      const response = await fetch(`/api/projects/${projectId}/rooms/${roomId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete room');
      }

      router.push(`/projects/${projectId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-muted-foreground">Loading room...</div>
      </div>
    );
  }

  if (error || !room) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-destructive">Error: {error || 'Room not found'}</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <Link href={`/projects/${projectId}`} className="inline-flex items-center text-muted-foreground hover:text-foreground mb-6">
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Project
      </Link>

      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">{room.name}</h1>
          <p className="text-muted-foreground">
            {room.project.name} • {room.roomType}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleDeleteRoom}>
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </Button>
          <Button onClick={() => router.push(`/studio?project=${projectId}&room=${roomId}`)}>
            <Edit className="h-4 w-4 mr-2" />
            Open in Studio
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Room Details</h2>
          <div className="space-y-3">
            <div>
              <span className="text-sm text-muted-foreground">Type:</span>
              <p className="font-medium">{room.roomType}</p>
            </div>
            {(room.width || room.height || room.length) && (
              <div>
                <span className="text-sm text-muted-foreground">Dimensions:</span>
                <p className="font-medium">
                  {room.width ? `${room.width}m` : '?'} × {room.height ? `${room.height}m` : '?'} × {room.length ? `${room.length}m` : '?'}
                </p>
              </div>
            )}
            <div>
              <span className="text-sm text-muted-foreground">Status:</span>
              <p className="font-medium">{room.isActive ? 'Active' : 'Inactive'}</p>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Created:</span>
              <p className="font-medium">{new Date(room.createdAt).toLocaleDateString()}</p>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Last Updated:</span>
              <p className="font-medium">{new Date(room.updatedAt).toLocaleDateString()}</p>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Design Data</h2>
          <div className="space-y-3">
            <div>
              <span className="text-sm text-muted-foreground">Layout Data:</span>
              <p className="font-medium">{room.layoutData ? 'Available' : 'Not set'}</p>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Scene Data:</span>
              <p className="font-medium">{room.sceneData ? 'Available' : 'Not set'}</p>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Vibe Spec:</span>
              <p className="font-medium">{room.vibeSpec ? 'Available' : 'Not set'}</p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
