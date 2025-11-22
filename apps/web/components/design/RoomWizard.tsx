'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ArrowLeft, ArrowRight, Check, Ruler, Layout, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';

// Define types locally for now to avoid build issues if package linking is tricky
// Ideally these would come from @3dix/types
type RoomType = 'living_room' | 'bedroom' | 'kitchen' | 'bathroom' | 'office' | 'dining_room' | 'other';

interface RoomConfig {
    name: string;
    type: RoomType;
    width: number;
    length: number;
    height: number;
    constraints: Record<string, any>;
}

const ROOM_TYPES: { value: RoomType; label: string; description: string }[] = [
    { value: 'living_room', label: 'Living Room', description: 'A space for relaxation and entertainment' },
    { value: 'bedroom', label: 'Bedroom', description: 'A personal space for rest and sleep' },
    { value: 'kitchen', label: 'Kitchen', description: 'For cooking and food preparation' },
    { value: 'dining_room', label: 'Dining Room', description: 'For shared meals and gathering' },
    { value: 'office', label: 'Office', description: 'A workspace for productivity' },
    { value: 'bathroom', label: 'Bathroom', description: 'Personal hygiene and care' },
    { value: 'other', label: 'Other', description: 'Custom room type' },
];

interface RoomWizardProps {
    projectId: string;
    onCancel: () => void;
}

/**
 * Render a three-step wizard UI for creating a new room within a project.
 *
 * The wizard collects basic details, lets the user choose a room type, and provides a review/constraint step.
 * On completion it POSTs the room configuration to the API and navigates to the created room page.
 *
 * @param projectId - Identifier of the project the new room will belong to; included in the creation request.
 * @param onCancel - Callback invoked when the user cancels the wizard (used on the first step or when closing).
 * @returns The RoomWizard component's JSX element.
 */
export function RoomWizard({ projectId, onCancel }: RoomWizardProps) {
    const router = useRouter();
    const [step, setStep] = useState(1);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [config, setConfig] = useState<RoomConfig>({
        name: '',
        type: 'living_room',
        width: 5.0,
        length: 5.0,
        height: 2.4,
        constraints: {},
    });

    const updateConfig = (updates: Partial<RoomConfig>) => {
        setConfig((prev) => ({ ...prev, ...updates }));
    };

    const handleSubmit = async () => {
        setLoading(true);
        setError(null);

        try {
            const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';
            const response = await fetch(`${baseUrl}/rooms/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...config,
                    project_id: projectId,
                }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to create room');
            }

            const room = await response.json();
            router.push(`/projects/${projectId}/rooms/${room._id}`);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
            setLoading(false);
        }
    };

    const nextStep = () => setStep((s) => Math.min(s + 1, 3));
    const prevStep = () => setStep((s) => Math.max(s - 1, 1));

    const renderStepIndicator = () => (
        <div className="flex items-center justify-center mb-8">
            {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center">
                    <div
                        className={cn(
                            "flex items-center justify-center w-8 h-8 rounded-full border-2 text-sm font-bold transition-colors",
                            step === i
                                ? "border-primary bg-primary text-primary-foreground"
                                : step > i
                                    ? "border-primary bg-primary text-primary-foreground"
                                    : "border-muted-foreground text-muted-foreground"
                        )}
                    >
                        {step > i ? <Check className="w-4 h-4" /> : i}
                    </div>
                    {i < 3 && (
                        <div
                            className={cn(
                                "w-16 h-0.5 mx-2 transition-colors",
                                step > i ? "bg-primary" : "bg-muted"
                            )}
                        />
                    )}
                </div>
            ))}
        </div>
    );

    return (
        <Card className="p-6 max-w-2xl mx-auto">
            <div className="mb-6 text-center">
                <h1 className="text-2xl font-bold">Create New Room</h1>
                <p className="text-muted-foreground">Step {step} of 3</p>
            </div>

            {renderStepIndicator()}

            <div className="min-h-[300px]">
                {step === 1 && (
                    <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                        <div className="flex items-center gap-2 mb-4">
                            <Ruler className="w-5 h-5 text-primary" />
                            <h2 className="text-xl font-semibold">Basic Details</h2>
                        </div>

                        <div>
                            <Label htmlFor="name">Room Name</Label>
                            <Input
                                id="name"
                                value={config.name}
                                onChange={(e) => updateConfig({ name: e.target.value })}
                                placeholder="e.g., Master Bedroom"
                                autoFocus
                            />
                        </div>

                        <div className="grid grid-cols-3 gap-4">
                            <div>
                                <Label htmlFor="width">Width (m)</Label>
                                <Input
                                    id="width"
                                    type="number"
                                    step="0.1"
                                    min="0.1"
                                    value={config.width}
                                    onChange={(e) => updateConfig({ width: parseFloat(e.target.value) })}
                                />
                            </div>
                            <div>
                                <Label htmlFor="length">Length (m)</Label>
                                <Input
                                    id="length"
                                    type="number"
                                    step="0.1"
                                    min="0.1"
                                    value={config.length}
                                    onChange={(e) => updateConfig({ length: parseFloat(e.target.value) })}
                                />
                            </div>
                            <div>
                                <Label htmlFor="height">Height (m)</Label>
                                <Input
                                    id="height"
                                    type="number"
                                    step="0.1"
                                    min="0.1"
                                    value={config.height}
                                    onChange={(e) => updateConfig({ height: parseFloat(e.target.value) })}
                                />
                            </div>
                        </div>
                    </div>
                )}

                {step === 2 && (
                    <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                        <div className="flex items-center gap-2 mb-4">
                            <Layout className="w-5 h-5 text-primary" />
                            <h2 className="text-xl font-semibold">Room Type</h2>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            {ROOM_TYPES.map((type) => (
                                <button
                                    key={type.value}
                                    onClick={() => updateConfig({ type: type.value })}
                                    className={cn(
                                        "p-4 rounded-lg border-2 text-left transition-all hover:border-primary/50",
                                        config.type === type.value
                                            ? "border-primary bg-primary/5"
                                            : "border-muted"
                                    )}
                                >
                                    <div className="font-semibold">{type.label}</div>
                                    <div className="text-sm text-muted-foreground">{type.description}</div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {step === 3 && (
                    <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                        <div className="flex items-center gap-2 mb-4">
                            <Settings className="w-5 h-5 text-primary" />
                            <h2 className="text-xl font-semibold">Review & Constraints</h2>
                        </div>

                        <div className="bg-muted/50 p-4 rounded-lg space-y-2">
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Name:</span>
                                <span className="font-medium">{config.name}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Type:</span>
                                <span className="font-medium capitalize">{config.type.replace('_', ' ')}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Dimensions:</span>
                                <span className="font-medium">{config.width}m x {config.length}m x {config.height}m</span>
                            </div>
                        </div>

                        <div className="p-4 border rounded-lg bg-yellow-50/50 dark:bg-yellow-900/10 border-yellow-200 dark:border-yellow-900/50">
                            <h3 className="font-medium mb-2">Constraint Configuration</h3>
                            <p className="text-sm text-muted-foreground">
                                Advanced constraint configuration will be available in the next update.
                                For now, default constraints for <strong>{config.type.replace('_', ' ')}</strong> will be applied.
                            </p>
                        </div>
                    </div>
                )}
            </div>

            {error && (
                <div className="mt-4 p-3 bg-destructive/10 text-destructive rounded-md text-sm">
                    {error}
                </div>
            )}

            <div className="flex justify-between mt-8 pt-4 border-t">
                <Button
                    variant="outline"
                    onClick={step === 1 ? onCancel : prevStep}
                    disabled={loading}
                >
                    {step === 1 ? 'Cancel' : 'Back'}
                </Button>

                {step < 3 ? (
                    <Button onClick={nextStep} disabled={!config.name}>
                        Next <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                ) : (
                    <Button onClick={handleSubmit} disabled={loading}>
                        {loading ? 'Creating...' : 'Create Room'}
                    </Button>
                )}
            </div>
        </Card>
    );
}