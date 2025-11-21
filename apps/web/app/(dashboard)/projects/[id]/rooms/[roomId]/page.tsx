'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { CustomObjectModal } from '@/components/design/CustomObjectModal';
import { ArrowLeft, Save, RotateCcw, Download, Wand2 } from 'lucide-react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { VibePanel, VibeFormState } from '@/components/design/VibePanel';
import { SceneHistory } from '@/components/design/SceneHistory';
import { JobProgress } from '@/components/jobs/JobProgress';
import { useJobPolling } from '@/hooks/useJobPolling';
import { SceneObject3D, SceneObject2D } from '@3dix/types';
import { MaskControlsState } from '@/components/design/MaskControls';
import { SemanticMapViewer } from '@/components/layout/SemanticMapViewer';
import { LayoutScene3D } from '@3dix/three';

// Dynamically import Canvas2D to avoid SSR issues with Konva
const Canvas2D = dynamic(
    () => import('@/components/design/Canvas2D').then((mod) => mod.Canvas2D),
    { ssr: false }
);
const LayoutScene3D = dynamic(
    () => import('@3dix/three').then((mod) => mod.LayoutScene3D),
    { ssr: false }
);

interface Room {
    _id: string;
    name: string;
    type: string;
    width: number;
    length: number;
    height: number;
    created_at: string;
    updated_at: string;
}

/**
 * Renders the room editor for editing room details, scene objects, and generating layouts.
 *
 * Manages room fetching, design state (2D/3D objects, vibe, history, masks, view modes, selection),
 * layout job lifecycle and polling, custom object replacement flow, and the UI for canvas,
 * semantic map, and 3D views alongside history and vibe controls.
 *
 * @returns A React element that renders the room editor page
 */
export default function RoomPage() {
    const params = useParams();
    const projectId = params.id as string;
    const roomId = params.roomId as string;

    const [room, setRoom] = useState<Room | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Design State
    const [objects, setObjects] = useState<SceneObject2D[]>([]);
    const [vibe, setVibe] = useState<VibeFormState>({
        prompt: '',
        keywords: [],
        style_sliders: {},
    });
    const [history, setHistory] = useState<any[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [jobId, setJobId] = useState<number | string | null>(null);
    const [lastRenderedJobId, setLastRenderedJobId] = useState<number | string | null>(null);
    const [maskControls, setMaskControls] = useState<MaskControlsState>({ maskType: 'none', maskUrl: '' });
    const [viewMode, setViewMode] = useState<'canvas' | 'semantic' | '3d'>('canvas');
    const [quality, setQuality] = useState<'low' | 'medium' | 'high'>('high');
    const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
    const [objects3D, setObjects3D] = useState<SceneObject3D[]>([]);
    const [isCustomObjectModalOpen, setIsCustomObjectModalOpen] = useState(false);

    const { job, loading: jobLoading, error: jobError } = useJobPolling(jobId, 3000);

    const handleCustomObjectUpload = async (imageUrl: string) => {
        if (!selectedObjectId || !room) return;

        try {
            const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';
            const response = await fetch(`${baseUrl}/rooms/${roomId}/objects/${selectedObjectId}/custom-mesh`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_url: imageUrl,
                }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to create custom mesh');
            }

            const data = await response.json();

            // Update 3D objects with the new mesh URL
            setObjects3D(prev => prev.map(obj =>
                obj.id === selectedObjectId
                    ? { ...obj, mesh_url: data.mesh_url }
                    : obj
            ));

            // Update 2D objects to indicate custom status (optional, e.g. change color)
            setObjects(prev => prev.map(obj =>
                obj.id === selectedObjectId
                    ? { ...obj, color: '#8b5cf6' } // Violet color to match the button
                    : obj
            ));

            addToHistory(`Replaced object ${selectedObjectId} with custom mesh`);
        } catch (err) {
            console.error('Failed to upload custom object:', err);
            setError(err instanceof Error ? err.message : 'Failed to upload custom object');
            throw err; // Re-throw to be caught by the modal
        }
    };

    const fetchRoom = useCallback(async () => {
        try {
            setLoading(true);
            const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';
            const response = await fetch(`${baseUrl}/rooms/${roomId}`);
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
    }, [roomId]);

    useEffect(() => {
        if (roomId) {
            fetchRoom();
        }
    }, [roomId, fetchRoom]);

    const handleObjectUpdate = (id: string, newProps: Partial<SceneObject2D>) => {
        setObjects((prev) =>
            prev.map((obj) => (obj.id === id ? { ...obj, ...newProps } : obj))
        );
    };

    const mapLayoutObjectsToScene = useCallback((layoutObjects: SceneObject3D[]): SceneObject2D[] => {
        return layoutObjects.map((obj) => ({
            id: obj.id || Math.random().toString(),
            type: obj.category || 'object',
            position: {
                x: Array.isArray(obj.position) ? obj.position[0] ?? 0 : 0,
                y: Array.isArray(obj.position) ? obj.position[2] ?? 0 : 0,
            },
            rotation: typeof obj.orientation === 'number' ? obj.orientation * 90 : 0,
            dimensions: {
                width: Array.isArray(obj.size) ? obj.size[0] ?? 1 : 1,
                depth: Array.isArray(obj.size) ? obj.size[2] ?? 1 : 1,
            },
            color: '#2563eb',
        }));
    }, []);

    const handleGenerate = async () => {
        if (!room) return;
        setIsGenerating(true);
        setError(null);

        try {
            const response = await fetch('/api/layout-jobs', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    room_id: Number(roomId),
                    vibe_spec: vibe,
                    constraints: {
                        arch_mask_url: maskControls.maskUrl || undefined,
                        mask_type: maskControls.maskType,
                    },
                }),
            });

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                const message = data?.error?.message || data?.error || 'Failed to create job';
                throw new Error(message);
            }

            const data = await response.json();
            setJobId(data.job_id);
            setLastRenderedJobId(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to create layout job');
            setIsGenerating(false);
        }
    };

    const addToHistory = (description: string) => {
        const newEntry = {
            id: Math.random().toString(),
            timestamp: new Date(),
            description,
        };
        setHistory((prev) => [newEntry, ...prev]);
    };

    const handleRestore = (historyId: string) => {
        console.log('Restore version:', historyId);
        // TODO: Implement restore logic
    };

    useEffect(() => {
        if (!jobId || !job) return;

        if (job.status === 'queued' || job.status === 'running') {
            setIsGenerating(true);
        } else {
            setIsGenerating(false);
        }

        if (job.status === 'completed' && job.result && jobId !== lastRenderedJobId) {
            const layoutObjects = (job.result.objects || []) as SceneObject3D[];
            if (layoutObjects.length > 0) {
                setObjects(mapLayoutObjectsToScene(layoutObjects));
                setObjects3D(layoutObjects);
                addToHistory(`Generated layout via job #${jobId}`);
                setSelectedObjectId(layoutObjects[0]?.id ?? null);
            }
            setLastRenderedJobId(jobId);
        }
    }, [job, jobId, lastRenderedJobId, mapLayoutObjectsToScene]);

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
        <div className="h-screen flex flex-col overflow-hidden">
            {/* Header */}
            <header className="border-b bg-background p-4 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-4">
                    <Link href={`/projects/${projectId}`} className="inline-flex items-center text-muted-foreground hover:text-foreground">
                        <ArrowLeft className="h-4 w-4 mr-2" />
                        Back
                    </Link>
                    <div>
                        <h1 className="text-xl font-bold">{room.name}</h1>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span className="capitalize">{room.type.replace('_', ' ')}</span>
                            <span>•</span>
                            <span>{room.width}m x {room.length}m x {room.height}m</span>
                        </div>
                    </div>
                </div>
                <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Reset
                    </Button>
                    <Button variant="outline" size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Export
                    </Button>
                    <Button size="sm">
                        <Save className="h-4 w-4 mr-2" />
                        Save
                    </Button>
                </div>
            </header>

            {/* Main Content */}
            <div className="flex-1 flex overflow-hidden">
                {/* Canvas Area */}
                <main className="flex-1 bg-slate-50 p-8 overflow-hidden flex items-center justify-center relative">
                    <div className="shadow-2xl rounded-lg overflow-hidden border bg-white relative w-full">
                        <div className="absolute top-4 right-4 z-10 flex gap-2">
                            <Button size="sm" variant={viewMode === 'canvas' ? 'default' : 'outline'} onClick={() => setViewMode('canvas')}>
                                Canvas
                            </Button>
                            <Button size="sm" variant={viewMode === 'semantic' ? 'default' : 'outline'} onClick={() => setViewMode('semantic')}>
                                Semantic Map
                            </Button>
                            <Button size="sm" variant={viewMode === '3d' ? 'default' : 'outline'} onClick={() => setViewMode('3d')}>
                                3D
                            </Button>
                            {viewMode === '3d' && (
                                <div className="flex gap-1">
                                    <Button size="sm" variant={quality === 'low' ? 'default' : 'outline'} onClick={() => setQuality('low')}>
                                        Low
                                    </Button>
                                    <Button size="sm" variant={quality === 'medium' ? 'default' : 'outline'} onClick={() => setQuality('medium')}>
                                        Med
                                    </Button>
                                    <Button size="sm" variant={quality === 'high' ? 'default' : 'outline'} onClick={() => setQuality('high')}>
                                        High
                                    </Button>
                                </div>
                            )}
                        </div>

                        {/* Custom Object Button */}
                        {selectedObjectId && viewMode === 'canvas' && (
                            <div className="absolute bottom-4 left-4 z-10">
                                <Button
                                    size="sm"
                                    onClick={() => setIsCustomObjectModalOpen(true)}
                                    className="bg-violet-600 hover:bg-violet-700 text-white shadow-lg"
                                >
                                    <Wand2 className="h-4 w-4 mr-2" />
                                    Replace with Custom Furniture
                                </Button>
                            </div>
                        )}

                        {viewMode === 'canvas' ? (
                            <Canvas2D
                                width={800}
                                height={600}
                                roomDimensions={{ width: room.width, length: room.length }}
                                objects={objects}
                                onObjectUpdate={handleObjectUpdate}
                                selectedObjectId={selectedObjectId}
                                onSelectObject={setSelectedObjectId}
                            />
                        ) : viewMode === 'semantic' ? (
                            <div className="p-4">
                                <SemanticMapViewer
                                    semanticMapUrl={(job?.result as any)?.semantic_map_png_url}
                                    objects={(job?.result?.objects as any) || []}
                                    roomWidth={room.width}
                                    roomLength={room.length}
                                    selectedId={selectedObjectId || undefined}
                                    onObjectClick={(id) => setSelectedObjectId(id)}
                                />
                            </div>
                        ) : (
                            <div className="h-[600px] w-[800px]">
                                <LayoutScene3D
                                    objects={objects3D}
                                    roomOutline={(job?.result as any)?.room_outline}
                                    selectedId={selectedObjectId || undefined}
                                    quality={quality}
                                    onSelect={(id) => setSelectedObjectId(id)}
                                />
                            </div>
                        )}
                    </div>
                </main>

                {/* Right Sidebar */}
                <aside className="w-80 border-l bg-background flex flex-col shrink-0">
                    <SceneHistory history={history} onRestore={handleRestore} />
                    <div className="flex-1 overflow-hidden space-y-4 p-4">
                        <JobProgress job={job} loading={jobLoading} />
                        {jobError && (
                            <div className="text-sm text-destructive">
                                {jobError}
                            </div>
                        )}
                        {error && (
                            <div className="text-sm text-destructive">
                                {error}
                            </div>
                        )}
                        <VibePanel
                            vibe={vibe}
                            onVibeUpdate={setVibe}
                            maskControls={maskControls}
                            onMaskChange={setMaskControls}
                            onGenerate={handleGenerate}
                            isGenerating={isGenerating}
                        />
                    </div>
                </aside>
            </div>

            <CustomObjectModal
                isOpen={isCustomObjectModalOpen}
                onClose={() => setIsCustomObjectModalOpen(false)}
                onUpload={handleCustomObjectUpload}
                objectCategory={objects.find(o => o.id === selectedObjectId)?.type || 'Furniture'}
            />
        </div>
    );
}