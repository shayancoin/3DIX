'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { LayoutCanvas } from '@/components/layout/LayoutCanvas';
import { VibePanel } from '@/components/vibe/VibePanel';
import { JobProgress } from '@/components/jobs/JobProgress';
import { useJobPolling } from '@/hooks/useJobPolling';
import { LayoutCanvasState, VibeSpec, RoomType, SceneObject2D } from '@3dix/types';

export default function StudioPage() {
  const searchParams = useSearchParams();
  const projectId = searchParams.get('project');
  const roomId = searchParams.get('room');

  const [roomType, setRoomType] = useState<RoomType>('kitchen');
  const [roomWidth, setRoomWidth] = useState(5);
  const [roomLength, setRoomLength] = useState(4);
  const [canvasState, setCanvasState] = useState<LayoutCanvasState | null>(null);
  const [vibeSpec, setVibeSpec] = useState<VibeSpec | null>(null);
  const [initialObjects, setInitialObjects] = useState<SceneObject2D[]>([]);
  const [currentJobId, setCurrentJobId] = useState<number | null>(null);
  const [roomIdNum, setRoomIdNum] = useState<number | null>(null);

  const { job, loading: jobLoading } = useJobPolling(currentJobId);

  // Load room data if roomId is provided
  useEffect(() => {
    if (roomId && projectId) {
      const roomIdNum = parseInt(roomId, 10);
      setRoomIdNum(roomIdNum);

      fetch(`/api/projects/${projectId}/rooms/${roomId}`)
        .then((res) => res.json())
        .then((room) => {
          setRoomType(room.roomType as RoomType);
          setRoomWidth(room.width || 5);
          setRoomLength(room.length || 4);
          if (room.layoutData?.objects) {
            setInitialObjects(room.layoutData.objects);
          }
          if (room.vibeSpec) {
            setVibeSpec(room.vibeSpec);
          }
        })
        .catch(console.error);

      // Load latest job for this room
      fetch(`/api/jobs?roomId=${roomIdNum}`)
        .then((res) => res.json())
        .then((jobs) => {
          if (jobs.length > 0) {
            const latestJob = jobs[0];
            if (latestJob.status === 'queued' || latestJob.status === 'running') {
              setCurrentJobId(latestJob.id);
            }
          }
        })
        .catch(console.error);
    }
  }, [roomId, projectId]);

  const handleVibeSpecChange = (newVibeSpec: VibeSpec) => {
    setVibeSpec(newVibeSpec);
  };

  const handleVibeSubmit = async (spec: VibeSpec) => {
    console.log('Generating layout with vibe spec:', spec);
  };

  const handleJobCreated = (jobId: number) => {
    setCurrentJobId(jobId);
  };

  // Update canvas when job completes
  useEffect(() => {
    if (job?.status === 'completed' && job.responseData?.objects) {
      // Convert 3D layout objects to 2D scene objects for canvas
      const sceneObjects = job.responseData.objects.map((obj: any, index: number) => ({
        id: obj.id || `obj-${index}`,
        category: obj.category,
        position: { x: obj.position[0], y: obj.position[2] }, // Use x and z for 2D
        size: { width: obj.size[0], height: obj.size[2] },
        boundingBox: {
          x: obj.position[0],
          y: obj.position[2],
          width: obj.size[0],
          height: obj.size[2],
        },
        label: obj.category,
      }));
      setInitialObjects(sceneObjects);
    }
  }, [job]);

  const handleCanvasStateChange = (state: LayoutCanvasState) => {
    setCanvasState(state);
    // TODO: Auto-save canvas state to room
  };

  return (
    <div className="flex flex-col h-screen w-full">
      <header className="p-4 border-b bg-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">3DIX Studio</h1>
            {projectId && roomId && (
              <p className="text-sm text-muted-foreground">
                Project: {projectId} • Room: {roomId}
              </p>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1 text-sm border rounded">Save</button>
            <button className="px-3 py-1 text-sm border rounded">Export</button>
          </div>
        </div>
      </header>
      <main className="flex-1 flex overflow-hidden">
        <div className="flex-1 flex flex-col min-w-0">
          {/* Job Progress */}
          {currentJobId && (
            <div className="p-4 border-b bg-white">
              <JobProgress job={job} loading={jobLoading} />
            </div>
          )}
          <LayoutCanvas
            initialObjects={initialObjects}
            roomWidth={roomWidth}
            roomLength={roomLength}
            onStateChange={handleCanvasStateChange}
          />
        </div>
        <div className="w-80 flex-shrink-0">
          <VibePanel
            initialVibeSpec={vibeSpec || undefined}
            roomType={roomType}
            roomId={roomIdNum || undefined}
            onVibeSpecChange={handleVibeSpecChange}
            onSubmit={handleVibeSubmit}
            onJobCreated={handleJobCreated}
          />
        </div>
      </main>
    </div>
  );
}
