'use client';

import { CanvasShell, SceneLayoutView } from '@3dix/three';

export default function StudioPage() {
    return (
        <div className="flex flex-col h-screen w-full">
            <header className="p-4 border-b">
                <h1 className="text-xl font-bold">3DIX Studio</h1>
            </header>
            <main className="flex-1 relative">
                <CanvasShell>
                    <SceneLayoutView />
                </CanvasShell>
            </main>
        </div>
    );
}
