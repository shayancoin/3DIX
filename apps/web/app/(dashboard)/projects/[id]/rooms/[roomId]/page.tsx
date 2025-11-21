'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Save, RotateCcw, Download } from 'lucide-react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { VibePanel } from '@/components/design/VibePanel';
import { SceneHistory } from '@/components/design/SceneHistory';
import { SceneObject2D, VibeSpec } from '@3dix/types';

// Dynamically import Canvas2D to avoid SSR issues with Konva
const Canvas2D = dynamic(
    () => import('@/components/design/Canvas2D').then((mod) => mod.Canvas2D),
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
 * Render the room editor page for a specific project room.
 *
 * Fetches room details from the current route parameters, manages editing state for
 * scene objects, generation vibe, and layout history, and provides UI for viewing
 * and modifying the room (canvas, vibe controls, history, and header actions).
 *
 * @returns A React element representing the room editor page
 */
export default function RoomPage() {
    const router = useRouter();
    const params = useParams();
    const projectId = params.id as string;
    const roomId = params.roomId as string;

    const [room, setRoom] = useState<Room | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Design State
    const [objects, setObjects] = useState<SceneObject2D[]>([]);
    const [vibe, setVibe] = useState<VibeSpec>({
        prompt: '',
        keywords: [],
        style_sliders: {},
    });
    const [history, setHistory] = useState<any[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);

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

    const handleGenerate = async () => {
        setIsGenerating(true);
        // TODO: Call generation API
        console.log('Generating layout with vibe:', vibe);

        // Simulate generation for now
        setTimeout(() => {
            const newObject: SceneObject2D = {
                id: Math.random().toString(),
                type: 'sofa',
                position: { x: room!.width / 2, y: room!.length / 2 },
                rotation: 0,
                dimensions: { width: 2, depth: 0.8 },
                color: '#e11d48',
            };
            setObjects((prev) => [...prev, newObject]);
            setIsGenerating(false);
            addToHistory('Generated new layout');
        }, 1500);
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
                    <div className="shadow-2xl rounded-lg overflow-hidden border bg-white">
                        <Canvas2D
                            width={800}
                            height={600}
                            roomDimensions={{ width: room.width, length: room.length }}
                            objects={objects}
                            onObjectUpdate={handleObjectUpdate}
                        />
                    </div>
                </main>

                {/* Right Sidebar */}
                <aside className="w-80 border-l bg-background flex flex-col shrink-0">
                    <SceneHistory history={history} onRestore={handleRestore} />
                    <div className="flex-1 overflow-hidden">
                        <VibePanel
                            vibe={vibe}
                            onVibeUpdate={setVibe}
                            onGenerate={handleGenerate}
                            isGenerating={isGenerating}
                        />
                    </div>
                </aside>
            </div>
        </div>
    );
}