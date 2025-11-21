'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Plus, ArrowLeft, Trash2, DoorOpen } from 'lucide-react';
import Link from 'next/link';

interface Room {
  _id: string;
  name: string;
  type: string;
  created_at: string;
  updated_at: string;
}

interface Project {
  _id: string;
  name: string;
  description: string | null;
  // rooms are fetched separately in our API design, but we can fetch them here
}

export default function ProjectDetailPage() {
  const router = useRouter();
  const params = useParams();
  const projectId = params.id as string;

  const [project, setProject] = useState<Project | null>(null);
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProjectAndRooms = useCallback(async () => {
    try {
      setLoading(true);
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';

      // Fetch Project
      const projectRes = await fetch(`${baseUrl}/projects/${projectId}`);
      if (!projectRes.ok) {
        throw new Error('Failed to fetch project');
      }
      const projectData = await projectRes.json();
      setProject(projectData);

      // Fetch Rooms
      const roomsRes = await fetch(`${baseUrl}/rooms/?project_id=${projectId}`);
      if (!roomsRes.ok) {
        throw new Error('Failed to fetch rooms');
      }
      const roomsData = await roomsRes.json();
      setRooms(roomsData);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    if (projectId) {
      fetchProjectAndRooms();
    }
  }, [projectId, fetchProjectAndRooms]);

  const handleDeleteProject = async () => {
    if (!confirm('Are you sure you want to delete this project? This will also delete all rooms in the project.')) {
      return;
    }

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';
      const response = await fetch(`${baseUrl}/projects/${projectId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete project');
      }

      router.push('/projects');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  };

  const handleCreateRoom = () => {
    router.push(`/projects/${projectId}/rooms/new`);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-muted-foreground">Loading project...</div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-destructive">Error: {error || 'Project not found'}</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <Link href="/projects" className="inline-flex items-center text-muted-foreground hover:text-foreground mb-6">
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Projects
      </Link>

      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">{project.name}</h1>
          {project.description && (
            <p className="text-muted-foreground">{project.description}</p>
          )}
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleDeleteProject}>
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </Button>
          <Button onClick={handleCreateRoom}>
            <Plus className="h-4 w-4 mr-2" />
            New Room
          </Button>
        </div>
      </div>

      <div className="mb-6">
        <h2 className="text-2xl font-semibold mb-4">Rooms ({rooms.length})</h2>

        {rooms.length === 0 ? (
          <Card className="p-12 text-center">
            <DoorOpen className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-xl font-semibold mb-2">No rooms yet</h3>
            <p className="text-muted-foreground mb-4">
              Create your first room to start designing
            </p>
            <Button onClick={handleCreateRoom}>
              <Plus className="mr-2 h-4 w-4" />
              Create Room
            </Button>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {rooms.map((room) => (
              <Link key={room._id} href={`/projects/${projectId}/rooms/${room._id}`}>
                <Card className="p-6 hover:shadow-lg transition-shadow cursor-pointer h-full">
                  <div className="flex items-start justify-between mb-4">
                    <DoorOpen className="h-8 w-8 text-primary" />
                    <span className="text-xs px-2 py-1 bg-secondary rounded">
                      {room.type}
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{room.name}</h3>
                  <div className="text-xs text-muted-foreground">
                    Updated {new Date(room.updated_at).toLocaleDateString()}
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
