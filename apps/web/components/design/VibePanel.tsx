'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { X, Wand2 } from 'lucide-react';
import { MaskControls, MaskControlsState } from './MaskControls';

export interface VibeFormState {
    prompt: string;
    keywords: string[];
    style_sliders: Record<string, number>;
}

interface VibePanelProps {
    vibe: VibeFormState;
    onVibeUpdate: (newVibe: VibeFormState) => void;
    maskControls: MaskControlsState;
    onMaskChange: (value: MaskControlsState) => void;
    onGenerate: () => void;
    isGenerating: boolean;
}

/**
 * Render a control panel for editing a VibeSpec and triggering layout generation.
 *
 * Provides controls to edit the vibe prompt, add/remove keyword tags, adjust style sliders
 * (minimalism, vintage, colorful), and trigger generation.
 *
 * @param vibe - Current VibeSpec state shown and edited by the panel
 * @param onVibeUpdate - Callback invoked with an updated VibeSpec when the user changes prompt, keywords, or sliders
 * @param onGenerate - Callback invoked when the user clicks the generate button
 * @param isGenerating - When true, disables the generate button and shows progress text
 * @returns The React element for the VibePanel UI
 */
export function VibePanel({ vibe, onVibeUpdate, maskControls, onMaskChange, onGenerate, isGenerating }: VibePanelProps) {
    const [keywordInput, setKeywordInput] = useState('');

    const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        onVibeUpdate({ ...vibe, prompt: e.target.value });
    };

    const handleKeywordKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && keywordInput.trim()) {
            e.preventDefault();
            if (!vibe.keywords.includes(keywordInput.trim())) {
                onVibeUpdate({
                    ...vibe,
                    keywords: [...vibe.keywords, keywordInput.trim()],
                });
            }
            setKeywordInput('');
        }
    };

    const removeKeyword = (keyword: string) => {
        onVibeUpdate({
            ...vibe,
            keywords: vibe.keywords.filter((k) => k !== keyword),
        });
    };

    const handleSliderChange = (key: string, value: number[]) => {
        onVibeUpdate({
            ...vibe,
            style_sliders: {
                ...vibe.style_sliders,
                [key]: value[0],
            },
        });
    };

    return (
        <div className="flex flex-col h-full gap-6 p-4 bg-background border-l">
            <div>
                <h2 className="text-lg font-semibold mb-1">Vibe Check</h2>
                <p className="text-sm text-muted-foreground">Define the style of your room.</p>
            </div>

            <div className="space-y-4">
                <div className="space-y-2">
                    <Label htmlFor="prompt">Prompt</Label>
                    <Textarea
                        id="prompt"
                        placeholder="A cozy living room with mid-century modern furniture and lots of plants..."
                        className="min-h-[100px] resize-none"
                        value={vibe.prompt}
                        onChange={handlePromptChange}
                    />
                </div>

                <div className="space-y-2">
                    <Label>Keywords</Label>
                    <div className="flex flex-wrap gap-2 mb-2">
                        {vibe.keywords.map((keyword) => (
                            <Badge key={keyword} variant="secondary" className="px-2 py-1">
                                {keyword}
                                <button
                                    onClick={() => removeKeyword(keyword)}
                                    className="ml-1 hover:text-destructive"
                                >
                                    <X className="h-3 w-3" />
                                </button>
                            </Badge>
                        ))}
                    </div>
                    <input
                        className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                        placeholder="Add keywords (press Enter)..."
                        value={keywordInput}
                        onChange={(e) => setKeywordInput(e.target.value)}
                        onKeyDown={handleKeywordKeyDown}
                    />
                </div>

                <div className="space-y-4 pt-4 border-t">
                    <Label>Style Sliders</Label>

                    <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                            <span>Minimalist</span>
                            <span>Maximalist</span>
                        </div>
                        <Slider
                            defaultValue={[0.5]}
                            max={1}
                            step={0.1}
                            value={[vibe.style_sliders['minimalism'] || 0.5]}
                            onValueChange={(val) => handleSliderChange('minimalism', val)}
                        />
                    </div>

                    <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                            <span>Modern</span>
                            <span>Vintage</span>
                        </div>
                        <Slider
                            defaultValue={[0.5]}
                            max={1}
                            step={0.1}
                            value={[vibe.style_sliders['vintage'] || 0.5]}
                            onValueChange={(val) => handleSliderChange('vintage', val)}
                        />
                    </div>

                    <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                            <span>Neutral</span>
                            <span>Colorful</span>
                        </div>
                        <Slider
                            defaultValue={[0.5]}
                            max={1}
                            step={0.1}
                            value={[vibe.style_sliders['colorful'] || 0.5]}
                            onValueChange={(val) => handleSliderChange('colorful', val)}
                        />
                    </div>
                </div>
            </div>

            <MaskControls value={maskControls} onChange={onMaskChange} />

            <div className="mt-auto">
                <Button
                    className="w-full"
                    size="lg"
                    onClick={onGenerate}
                    disabled={isGenerating}
                >
                    <Wand2 className="mr-2 h-4 w-4" />
                    {isGenerating ? 'Generating...' : 'Generate Layout'}
                </Button>
            </div>
        </div>
    );
}
