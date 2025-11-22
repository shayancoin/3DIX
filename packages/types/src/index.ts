// Room Types
export type RoomType = 'bedroom' | 'kitchen' | 'bathroom' | 'closet' | 'living_room' | 'dining_room' | 'office' | 'other';

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
  type: string; // e.g., 'sofa', 'table', 'chair'
  category: string; // e.g., 'sofa', 'table', 'chair' - same as type for now, but aligns with 3D
  position: { x: number; y: number };
  rotation: number; // in degrees
  dimensions: { width: number; depth: number };
  color?: string; // for placeholder visualization
  label?: string;
  metadata?: Record<string, any>;
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

export interface SceneObject3D {
  id: string;
  category: string;
  position: [number, number, number]; // metres
  size: [number, number, number];
  orientation: 0 | 1 | 2 | 3; // multiples of 90 degrees
  mesh_url?: string;
  metadata?: Record<string, any>;
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

// Constraint validation metadata
export type ConstraintSeverity = 'info' | 'warning' | 'error';

export interface ConstraintViolation {
  id: string;
  constraint_type: string;
  message: string;
  severity: ConstraintSeverity;
  metric_value: number;
  threshold: number;
  unit?: string | null;
  normalized_violation: number;
  object_ids: string[];
}

export interface ConstraintValidation {
  satisfied: boolean;
  max_violation: number;
  violations: ConstraintViolation[];
}

// Layout Request/Response Types
export interface LayoutRequest {
  room_type: RoomType;
  room_config?: any;
  arch_mask_url?: string;
  mask_type?: 'none' | 'floor' | 'arch';
  vibe_spec: VibeSpec;
  seed?: number;
}

export interface LayoutResponse {
  semantic_map_png_url?: string;
  objects: SceneObject3D[];
  world_scale: number; // meters per pixel or similar
  room_outline?: [number, number][];
  metadata?: Record<string, any>;
  constraint_validation?: ConstraintValidation | null;
}

export type GenerationJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface LayoutJobStatusResponse {
  job_id: string | number;
  status: GenerationJobStatus;
  result: LayoutResponse | null;
  error: string | null;
  progress: number | null;
  progress_message: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  updated_at?: string | null;
}

// Export room configuration types
export * from './room-configs';
