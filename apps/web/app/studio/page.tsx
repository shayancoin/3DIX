'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { LayoutCanvas } from '@/components/layout/LayoutCanvas';
import { VibePanel } from '@/components/vibe/VibePanel';
import { JobProgress } from '@/components/jobs/JobProgress';
import { SemanticMapViewer } from '@/components/layout/SemanticMapViewer';
import { CanvasShell, LayoutScene3D } from '@3dix/three';
import { useJobPolling } from '@/hooks/useJobPolling';
import { LayoutCanvasState, VibeSpec, RoomType, SceneObject2D, LayoutObject } from '@3dix/types';

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
  const [layoutObjects, setLayoutObjects] = useState<LayoutObject[]>([]);
  const [semanticMapUrl, setSemanticMapUrl] = useState<string | undefined>();
  const [selectedLayoutObjectId, setSelectedLayoutObjectId] = useState<string | undefined>();
  const [selectedCanvasObjectId, setSelectedCanvasObjectId] = useState<string | undefined>();
  const [viewMode, setViewMode] = useState<'canvas' | 'semantic' | '3d'>('canvas');
  const [meshQuality, setMeshQuality] = useState<'low' | 'medium' | 'high'>('high');
  const [useMeshes, setUseMeshes] = useState(true);
  
  // Sync selection between 2D and 3D views
  const handleObjectSelect = useCallback((objectId: string | undefined, source: '2d' | '3d' | 'semantic') => {
    if (source === '2d') {
      setSelectedCanvasObjectId(objectId);
      // Find corresponding layout object
      if (objectId) {
        const canvasObj = canvasState?.objects.find((o) => o.id === objectId);
        if (canvasObj) {
          // Find matching layout object by category and approximate position
          const layoutObj = layoutObjects.find((lo) => {
            const dx = Math.abs(lo.position[0] - canvasObj.position.x);
            const dz = Math.abs(lo.position[2] - canvasObj.position.y);
            return lo.category === canvasObj.category && dx < 0.5 && dz < 0.5;
          });
          if (layoutObj) {
            setSelectedLayoutObjectId(layoutObj.id);
          }
        }
      } else {
        setSelectedLayoutObjectId(undefined);
      }
    } else if (source === '3d' || source === 'semantic') {
      setSelectedLayoutObjectId(objectId);
      // Find corresponding canvas object
      if (objectId) {
        const layoutObj = layoutObjects.find((lo) => lo.id === objectId);
        if (layoutObj) {
          const canvasObj = canvasState?.objects.find((co) => {
            const dx = Math.abs(co.position.x - layoutObj.position[0]);
            const dy = Math.abs(co.position.y - layoutObj.position[2]);
            return co.category === layoutObj.category && dx < 0.5 && dy < 0.5;
          });
          if (canvasObj) {
            setSelectedCanvasObjectId(canvasObj.id);
          }
        }
      } else {
        setSelectedCanvasObjectId(undefined);
      }
    }
  }, [canvasState, layoutObjects]);

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

  // Update canvas and semantic map when job completes
  useEffect(() => {
    if (job?.status === 'completed' && job.responseData) {
      const responseData = job.responseData;
      
      // Store layout objects
      if (responseData.objects) {
        setLayoutObjects(responseData.objects);
        
        // Convert 3D layout objects to 2D scene objects for canvas
        const sceneObjects = responseData.objects.map((obj: any, index: number) => ({
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

      // Store semantic map
      if (responseData.semanticMap || responseData.mask) {
        setSemanticMapUrl(responseData.semanticMap || responseData.mask);
      }
    }
  }, [job]);

  const handleCanvasStateChange = (state: LayoutCanvasState) => {
    setCanvasState(state);
    // Sync selection
    if (state.selectedObjectId !== selectedCanvasObjectId) {
      handleObjectSelect(state.selectedObjectId, '2d');
    }
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
          
          {/* View Mode Toggle */}
          <div className="p-2 border-b bg-white flex gap-2">
            <button
              onClick={() => setViewMode('canvas')}
              className={`px-3 py-1 text-sm rounded ${
                viewMode === 'canvas'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-gray-100 hover:bg-gray-200'
              }`}
            >
              2D Canvas
            </button>
            <button
              onClick={() => setViewMode('semantic')}
              className={`px-3 py-1 text-sm rounded ${
                viewMode === 'semantic'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-gray-100 hover:bg-gray-200'
              }`}
            >
              Semantic Map
            </button>
            <button
              onClick={() => setViewMode('3d')}
              className={`px-3 py-1 text-sm rounded ${
                viewMode === '3d'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-gray-100 hover:bg-gray-200'
              }`}
            >
              3D View
            </button>
          </div>

          {/* Canvas, Semantic Map, or 3D View */}
          {viewMode === 'canvas' ? (
            <LayoutCanvas
              initialObjects={initialObjects}
              roomWidth={roomWidth}
              roomLength={roomLength}
              onStateChange={handleCanvasStateChange}
              selectedObjectId={selectedCanvasObjectId}
              onObjectSelect={(id) => handleObjectSelect(id, '2d')}
            />
          ) : viewMode === 'semantic' ? (
            <div className="flex-1 relative min-h-0 p-4">
              <SemanticMapViewer
                semanticMapUrl={semanticMapUrl}
                objects={layoutObjects}
                roomWidth={roomWidth}
                roomLength={roomLength}
                onObjectClick={(id) => handleObjectSelect(id, 'semantic')}
                selectedObjectId={selectedLayoutObjectId}
              />
            </div>
          ) : (
            <div className="flex-1 relative min-h-0 flex flex-col">
              {/* Quality Controls */}
              <div className="p-2 border-b bg-white flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={useMeshes}
                    onChange={(e) => setUseMeshes(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span>Use 3D Meshes</span>
                </label>
                {useMeshes && (
                  <div className="flex items-center gap-2">
                    <span className="text-sm">Quality:</span>
                    <select
                      value={meshQuality}
                      onChange={(e) => setMeshQuality(e.target.value as 'low' | 'medium' | 'high')}
                      className="px-2 py-1 text-sm border rounded"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>
                )}
              </div>
              <div className="flex-1 relative min-h-0">
                <CanvasShell>
                  <LayoutScene3D
                    objects={layoutObjects}
                    roomWidth={roomWidth}
                    roomLength={roomLength}
                    selectedObjectId={selectedLayoutObjectId}
                    onObjectClick={(id) => handleObjectSelect(id, '3d')}
                    showGrid={true}
                    showLabels={true}
                    meshQuality={meshQuality}
                    useMeshes={useMeshes}
                  />
                </CanvasShell>
              </div>
            </div>
          )}
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
