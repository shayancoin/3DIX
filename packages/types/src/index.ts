// Room Types
export const ROOM_TYPES = [
  'bedroom',
  'kitchen',
  'bathroom',
  'closet',
  'living',
  'dining',
  'custom',
] as const;

export type RoomType = (typeof ROOM_TYPES)[number];

// Vibe Specification Types
export interface VibePrompt {
  text: string;
  referenceImageUrl?: string;
  roomType: RoomType;
}

export interface VibeTag {
  id: string;
  label: string;
  category: 'style' | 'mood' | 'color' | 'material' | 'era';
  weight?: number; // 0-1, default 0.5
}

export interface VibeSlider {
  id: string;
  label: string;
  min: number;
  max: number;
  value: number;
  step?: number;
}

export interface VibeSpec {
  prompt: VibePrompt;
  tags: VibeTag[];
  sliders: VibeSlider[];
  metadata?: {
    createdAt?: string;
    updatedAt?: string;
  };
}

// 2D Scene Object Types
export interface Point2D {
  x: number;
  y: number;
}

export interface BoundingBox2D {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SceneObject2D {
  id: string;
  category: string; // e.g., 'refrigerator', 'sink', 'cabinet'
  label?: string;
  position: Point2D;
  size: { width: number; height: number };
  rotation?: number; // degrees
  boundingBox: BoundingBox2D;
  color?: string; // hex color for visualization
  metadata?: {
    confidence?: number;
    modelId?: string;
    [key: string]: any;
  };
}

// 3D Scene Object Types (for future use)
export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface LayoutObject {
  id: string;
  category: string;
  position: [number, number, number];
  size: [number, number, number];
  orientation: number; // radians
  metadata?: {
    assetId?: string;
    modelUrl?: string;
    assetUrl?: string;
    textureUrl?: string;
    assetQuality?: string;
    customMeshData?: string; // Base64 encoded custom mesh
    isCustom?: boolean; // Whether this is a custom reconstructed object
    [key: string]: any;
  };
}

// Layout Canvas Types
export interface CanvasViewport {
  x: number;
  y: number;
  zoom: number;
}

export interface LayoutCanvasState {
  objects: SceneObject2D[];
  viewport: CanvasViewport;
  selectedObjectId?: string;
  history: SceneHistoryEntry[];
  historyIndex: number;
}

// Scene History Types
export interface SceneHistoryEntry {
  id: string;
  timestamp: string;
  action: 'create' | 'update' | 'delete' | 'move' | 'rotate' | 'resize';
  objectId: string;
  previousState?: Partial<SceneObject2D>;
  newState?: Partial<SceneObject2D>;
  snapshot?: {
    objects: SceneObject2D[];
    viewport: CanvasViewport;
  };
}

// Layout Request/Response Types
export interface LayoutRequest {
  roomId: string;
  vibeSpec: VibeSpec;
  constraints?: {
    roomDimensions?: {
      width: number;
      height: number;
      length: number;
    };
    existingObjects?: SceneObject2D[];
    maskType?: 'none' | 'room_boundary' | 'wall_mask' | 'door_window_mask';
    maskImage?: string; // base64 encoded image
    assetQuality?: 'low' | 'medium' | 'high'; // Quality level for 3D asset retrieval
  };
}

export interface LayoutResponse {
  jobId: string;
  status: GenerationJobStatus;
  mask?: string; // base64 or url
  objects: LayoutObject[];
  semanticMap?: string; // base64 encoded semantic segmentation map
  metadata?: {
    processingTime?: number;
    modelVersion?: string;
    [key: string]: any;
  };
}

export type GenerationJobStatus = 'queued' | 'running' | 'completed' | 'failed';

// Export all types
export type {
  VibePrompt,
  VibeTag,
  VibeSlider,
  VibeSpec,
  Point2D,
  BoundingBox2D,
  SceneObject2D,
  Point3D,
  LayoutObject,
  CanvasViewport,
  LayoutCanvasState,
  SceneHistoryEntry,
  LayoutRequest,
  LayoutResponse,
  GenerationJobStatus,
};

// Export room configuration types
export * from './room-configs';
