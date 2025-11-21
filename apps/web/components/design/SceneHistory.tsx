'use client';

import { ScrollArea } from '@/components/ui/scroll-area';
import { Clock, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface HistoryItem {
    id: string;
    timestamp: Date;
    description: string;
}

interface SceneHistoryProps {
    history: HistoryItem[];
    onRestore: (id: string) => void;
}

export function SceneHistory({ history, onRestore }: SceneHistoryProps) {
    return (
        <div className="border-b p-4">
            <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-muted-foreground">
                <Clock className="h-4 w-4" />
                <span>History</span>
            </div>
            <ScrollArea className="h-[100px]">
                {history.length === 0 ? (
                    <div className="text-xs text-muted-foreground italic">No history yet</div>
                ) : (
                    <div className="space-y-2">
                        {history.map((item) => (
                            <div key={item.id} className="flex items-center justify-between text-sm group">
                                <span className="text-muted-foreground truncate max-w-[150px]">
                                    {item.description}
                                </span>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-6 w-6 opacity-0 group-hover:opacity-100"
                                    onClick={() => onRestore(item.id)}
                                    title="Restore this version"
                                >
                                    <RotateCcw className="h-3 w-3" />
                                </Button>
                            </div>
                        ))}
                    </div>
                )}
            </ScrollArea>
        </div>
    );
}
