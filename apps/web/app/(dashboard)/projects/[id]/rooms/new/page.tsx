'use client';

import { useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function NewRoomPage() {
    const router = useRouter();
    const params = useParams();
    const projectId = params.id as string;

    const [name, setName] = useState('');
    const [type, setType] = useState('living_room');
    const [width, setWidth] = useState(5.0);
    const [length, setLength] = useState(5.0);
    const [height, setHeight] = useState(2.4);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
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
                    name,
                    type,
                    width,
                    length,
                    height,
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
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto p-6 max-w-2xl">
            <Link href={`/projects/${projectId}`} className="inline-flex items-center text-muted-foreground hover:text-foreground mb-6">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Project
            </Link>

            <Card className="p-6">
                <h1 className="text-3xl font-bold mb-6">Create New Room</h1>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <Label htmlFor="name">Room Name *</Label>
                        <Input
                            id="name"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            required
                            placeholder="Main Living Room"
                        />
                    </div>

                    <div>
                        <Label htmlFor="type">Room Type</Label>
                        <Select value={type} onValueChange={setType}>
                            <SelectTrigger>
                                <SelectValue placeholder="Select room type" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="living_room">Living Room</SelectItem>
                                <SelectItem value="bedroom">Bedroom</SelectItem>
                                <SelectItem value="kitchen">Kitchen</SelectItem>
                                <SelectItem value="bathroom">Bathroom</SelectItem>
                                <SelectItem value="office">Office</SelectItem>
                                <SelectItem value="dining_room">Dining Room</SelectItem>
                                <SelectItem value="other">Other</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                        <div>
                            <Label htmlFor="width">Width (m)</Label>
                            <Input
                                id="width"
                                type="number"
                                step="0.1"
                                min="0.1"
                                value={width}
                                onChange={(e) => setWidth(parseFloat(e.target.value))}
                                required
                            />
                        </div>
                        <div>
                            <Label htmlFor="length">Length (m)</Label>
                            <Input
                                id="length"
                                type="number"
                                step="0.1"
                                min="0.1"
                                value={length}
                                onChange={(e) => setLength(parseFloat(e.target.value))}
                                required
                            />
                        </div>
                        <div>
                            <Label htmlFor="height">Height (m)</Label>
                            <Input
                                id="height"
                                type="number"
                                step="0.1"
                                min="0.1"
                                value={height}
                                onChange={(e) => setHeight(parseFloat(e.target.value))}
                                required
                            />
                        </div>
                    </div>

                    {error && (
                        <div className="text-destructive text-sm">{error}</div>
                    )}

                    <div className="flex gap-4 pt-4">
                        <Button type="submit" disabled={loading}>
                            {loading ? 'Creating...' : 'Create Room'}
                        </Button>
                        <Button
                            type="button"
                            variant="outline"
                            onClick={() => router.back()}
                        >
                            Cancel
                        </Button>
                    </div>
                </form>
            </Card>
        </div>
    );
}
