import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { X, Upload, Loader2 } from 'lucide-react';

interface CustomObjectModalProps {
    isOpen: boolean;
    onClose: () => void;
    onUpload: (imageUrl: string) => Promise<void>;
    objectCategory: string;
}

export function CustomObjectModal({ isOpen, onClose, onUpload, objectCategory }: CustomObjectModalProps) {
    const [imageUrl, setImageUrl] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    if (!isOpen) return null;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!imageUrl) return;

        setLoading(true);
        setError(null);

        try {
            await onUpload(imageUrl);
            onClose();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to process image');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-background w-full max-w-md rounded-lg shadow-lg border p-6 relative animate-in fade-in zoom-in-95 duration-200">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-muted-foreground hover:text-foreground"
                >
                    <X className="h-4 w-4" />
                </button>

                <h2 className="text-xl font-semibold mb-4">Replace with Custom Furniture</h2>
                <p className="text-sm text-muted-foreground mb-6">
                    Upload an image of your {objectCategory} to generate a 3D model.
                </p>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <Label htmlFor="image-url">Image URL</Label>
                        <Input
                            id="image-url"
                            value={imageUrl}
                            onChange={(e) => setImageUrl(e.target.value)}
                            placeholder="https://example.com/my-chair.jpg"
                            required
                        />
                    </div>

                    {error && (
                        <div className="text-sm text-destructive bg-destructive/10 p-2 rounded">
                            {error}
                        </div>
                    )}

                    <div className="flex justify-end gap-2 pt-2">
                        <Button type="button" variant="outline" onClick={onClose} disabled={loading}>
                            Cancel
                        </Button>
                        <Button type="submit" disabled={loading || !imageUrl}>
                            {loading ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Processing...
                                </>
                            ) : (
                                <>
                                    <Upload className="mr-2 h-4 w-4" />
                                    Generate 3D Model
                                </>
                            )}
                        </Button>
                    </div>
                </form>
            </div>
        </div>
    );
}
