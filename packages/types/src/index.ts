export type RoomType = 'bedroom' | 'kitchen' | 'bathroom' | 'closet' | 'custom';

export interface VibePrompt {
  text: string;
  referenceImageUrl?: string;
  roomType: RoomType;
}

export interface LayoutObject {
  category: string;
  position: [number, number, number];
  size: [number, number, number];
  orientation: number;
}

export interface LayoutRequest {
  prompt: VibePrompt;
  constraints?: any;
}

export interface LayoutResponse {
  mask: string; // base64 or url
  objects: LayoutObject[];
}

export type GenerationJobStatus = 'queued' | 'running' | 'completed' | 'failed';
