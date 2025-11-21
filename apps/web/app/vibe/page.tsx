'use client';

import { ChatPanel } from '@3dix/ui';

export default function VibePage() {
    return (
        <div className="flex h-screen w-full">
            <div className="flex-1 flex items-center justify-center bg-gray-50">
                <h1 className="text-2xl text-gray-400">Vibe Coding Preview</h1>
            </div>
            <ChatPanel />
        </div>
    );
}
