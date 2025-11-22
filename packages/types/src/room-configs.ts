/**
 * Domain-Specific Configuration Schemas for Room Types
 * Defines constraints, required objects, and validation rules for each room type
 */

import { RoomType, LayoutObject } from './index';

/**
 * Object category definitions with constraints
 */
export interface ObjectCategory {
  id: string;
  name: string;
  required: boolean;
  minCount?: number;
  maxCount?: number;
  minSize?: [number, number, number];
  maxSize?: [number, number, number];
  allowedPositions?: 'wall' | 'center' | 'corner' | 'any';
  spacing?: {
    minDistance?: number; // Minimum distance from other objects
    clearance?: number; // Required clearance around object
  };
  dependencies?: string[]; // IDs of categories that must exist
  conflicts?: string[]; // IDs of categories that cannot coexist
}

/**
 * Room type configuration schema
 */
export interface RoomTypeConfig {
  roomType: RoomType;
  name: string;
  description: string;
  defaultDimensions: {
    width: number;
    length: number;
    height: number;
  };
  categories: ObjectCategory[];
  constraints: {
    minObjects?: number;
    maxObjects?: number;
    requiredCategories?: string[]; // Category IDs that must be present
    layoutRules?: LayoutRule[];
  };
  zones?: RoomZone[]; // Functional zones within the room
}

/**
 * Layout rules for room organization
 */
export interface LayoutRule {
  id: string;
  name: string;
  type: 'proximity' | 'alignment' | 'orientation' | 'spacing' | 'accessibility';
  description: string;
  categoryIds: string[]; // Categories this rule applies to
  parameters: {
    [key: string]: any;
  };
  priority: 'required' | 'recommended' | 'optional';
}

/**
 * Functional zones within a room
 */
export interface RoomZone {
  id: string;
  name: string;
  bounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  allowedCategories?: string[]; // Categories allowed in this zone
  preferredCategories?: string[]; // Categories preferred in this zone
}

/**
 * Room type configurations
 */
export const ROOM_TYPE_CONFIGS: Record<RoomType, RoomTypeConfig> = {
  kitchen: {
    roomType: 'kitchen',
    name: 'Kitchen',
    description: 'Kitchen layout with appliances and storage',
    defaultDimensions: { width: 4, length: 3, height: 2.5 },
    categories: [
      {
        id: 'refrigerator',
        name: 'Refrigerator',
        required: true,
        minCount: 1,
        maxCount: 1,
        minSize: [0.6, 1.5, 0.6],
        maxSize: [1.0, 2.0, 0.8],
        allowedPositions: 'wall',
        spacing: { clearance: 0.1 },
      },
      {
        id: 'sink',
        name: 'Sink',
        required: true,
        minCount: 1,
        maxCount: 2,
        minSize: [0.5, 0.2, 0.5],
        maxSize: [1.2, 0.3, 0.8],
        allowedPositions: 'wall',
        dependencies: ['counter'],
      },
      {
        id: 'stove',
        name: 'Stove',
        required: true,
        minCount: 1,
        maxCount: 1,
        minSize: [0.6, 0.9, 0.6],
        maxSize: [0.9, 1.0, 0.8],
        allowedPositions: 'wall',
        spacing: { minDistance: 0.3, clearance: 0.2 },
      },
      {
        id: 'cabinet',
        name: 'Cabinet',
        required: false,
        minCount: 0,
        maxCount: 10,
        minSize: [0.3, 0.4, 0.3],
        maxSize: [1.5, 0.9, 0.6],
        allowedPositions: 'wall',
      },
      {
        id: 'counter',
        name: 'Counter',
        required: true,
        minCount: 1,
        maxCount: 5,
        minSize: [0.6, 0.9, 0.3],
        maxSize: [3.0, 1.0, 0.6],
        allowedPositions: 'wall',
      },
      {
        id: 'dishwasher',
        name: 'Dishwasher',
        required: false,
        minCount: 0,
        maxCount: 1,
        minSize: [0.6, 0.8, 0.6],
        maxSize: [0.6, 0.9, 0.6],
        allowedPositions: 'wall',
        dependencies: ['sink'],
      },
    ],
    constraints: {
      minObjects: 3,
      maxObjects: 15,
      requiredCategories: ['refrigerator', 'sink', 'stove'],
      layoutRules: [
        {
          id: 'kitchen_triangle',
          name: 'Kitchen Work Triangle',
          type: 'proximity',
          description: 'Refrigerator, sink, and stove should form a triangle',
          categoryIds: ['refrigerator', 'sink', 'stove'],
          parameters: { maxDistance: 2.5 },
          priority: 'recommended',
        },
        {
          id: 'sink_counter_proximity',
          name: 'Sink-Counter Proximity',
          type: 'proximity',
          description: 'Sink should be adjacent to counter space',
          categoryIds: ['sink', 'counter'],
          parameters: { maxDistance: 0.5 },
          priority: 'required',
        },
      ],
    },
    zones: [
      {
        id: 'cooking_zone',
        name: 'Cooking Zone',
        bounds: { x: 0, y: 0, width: 2, height: 3 },
        preferredCategories: ['stove', 'counter'],
      },
      {
        id: 'prep_zone',
        name: 'Prep Zone',
        bounds: { x: 1, y: 0, width: 2, height: 3 },
        preferredCategories: ['counter', 'sink'],
      },
      {
        id: 'storage_zone',
        name: 'Storage Zone',
        bounds: { x: 0, y: 0, width: 4, height: 1 },
        preferredCategories: ['refrigerator', 'cabinet'],
      },
    ],
  },
  bedroom: {
    roomType: 'bedroom',
    name: 'Bedroom',
    description: 'Bedroom layout with bed and furniture',
    defaultDimensions: { width: 4, length: 3.5, height: 2.5 },
    categories: [
      {
        id: 'bed',
        name: 'Bed',
        required: true,
        minCount: 1,
        maxCount: 2,
        minSize: [1.4, 0.4, 2.0],
        maxSize: [2.0, 0.6, 2.2],
        allowedPositions: 'any',
        spacing: { clearance: 0.5 },
      },
      {
        id: 'dresser',
        name: 'Dresser',
        required: false,
        minCount: 0,
        maxCount: 2,
        minSize: [0.8, 0.6, 0.4],
        maxSize: [1.5, 1.2, 0.6],
        allowedPositions: 'wall',
      },
      {
        id: 'nightstand',
        name: 'Nightstand',
        required: false,
        minCount: 0,
        maxCount: 2,
        minSize: [0.4, 0.5, 0.4],
        maxSize: [0.6, 0.7, 0.5],
        allowedPositions: 'any',
        dependencies: ['bed'],
      },
      {
        id: 'closet',
        name: 'Closet',
        required: false,
        minCount: 0,
        maxCount: 1,
        minSize: [1.0, 2.0, 0.6],
        maxSize: [2.0, 2.5, 0.8],
        allowedPositions: 'wall',
      },
    ],
    constraints: {
      minObjects: 1,
      maxObjects: 8,
      requiredCategories: ['bed'],
      layoutRules: [
        {
          id: 'bed_nightstand_proximity',
          name: 'Bed-Nightstand Proximity',
          type: 'proximity',
          description: 'Nightstands should be adjacent to bed',
          categoryIds: ['bed', 'nightstand'],
          parameters: { maxDistance: 0.3 },
          priority: 'recommended',
        },
        {
          id: 'bed_accessibility',
          name: 'Bed Accessibility',
          type: 'accessibility',
          description: 'Bed should have clear access from at least one side',
          categoryIds: ['bed'],
          parameters: { minClearance: 0.5 },
          priority: 'required',
        },
      ],
    },
    zones: [
      {
        id: 'sleeping_zone',
        name: 'Sleeping Zone',
        bounds: { x: 1, y: 1, width: 2, height: 2.5 },
        preferredCategories: ['bed', 'nightstand'],
      },
      {
        id: 'storage_zone',
        name: 'Storage Zone',
        bounds: { x: 0, y: 0, width: 4, height: 1 },
        preferredCategories: ['dresser', 'closet'],
      },
    ],
  },
  bathroom: {
    roomType: 'bathroom',
    name: 'Bathroom',
    description: 'Bathroom layout with fixtures',
    defaultDimensions: { width: 2.5, length: 2, height: 2.5 },
    categories: [
      {
        id: 'toilet',
        name: 'Toilet',
        required: true,
        minCount: 1,
        maxCount: 1,
        minSize: [0.4, 0.4, 0.7],
        maxSize: [0.5, 0.5, 0.8],
        allowedPositions: 'wall',
        spacing: { clearance: 0.3 },
      },
      {
        id: 'sink',
        name: 'Sink',
        required: true,
        minCount: 1,
        maxCount: 2,
        minSize: [0.5, 0.2, 0.4],
        maxSize: [0.8, 0.3, 0.6],
        allowedPositions: 'wall',
      },
      {
        id: 'shower',
        name: 'Shower',
        required: false,
        minCount: 0,
        maxCount: 1,
        minSize: [0.8, 0.8, 2.0],
        maxSize: [1.2, 1.2, 2.5],
        allowedPositions: 'corner',
        spacing: { clearance: 0.2 },
      },
      {
        id: 'bathtub',
        name: 'Bathtub',
        required: false,
        minCount: 0,
        maxCount: 1,
        minSize: [1.5, 0.6, 0.7],
        maxSize: [1.8, 0.7, 0.8],
        allowedPositions: 'wall',
        conflicts: ['shower'],
      },
    ],
    constraints: {
      minObjects: 2,
      maxObjects: 6,
      requiredCategories: ['toilet', 'sink'],
      layoutRules: [
        {
          id: 'toilet_privacy',
          name: 'Toilet Privacy',
          type: 'spacing',
          description: 'Toilet should have privacy clearance',
          categoryIds: ['toilet'],
          parameters: { minDistance: 0.3 },
          priority: 'required',
        },
      ],
    },
    zones: [
      {
        id: 'wet_zone',
        name: 'Wet Zone',
        bounds: { x: 0, y: 0, width: 2.5, height: 1.5 },
        preferredCategories: ['shower', 'bathtub'],
      },
      {
        id: 'dry_zone',
        name: 'Dry Zone',
        bounds: { x: 0, y: 1.5, width: 2.5, height: 0.5 },
        preferredCategories: ['toilet', 'sink'],
      },
    ],
  },
  living_room: {
    roomType: 'living_room',
    name: 'Living Room',
    description: 'Living room layout with seating and entertainment',
    defaultDimensions: { width: 5, length: 4, height: 2.5 },
    categories: [
      {
        id: 'sofa',
        name: 'Sofa',
        required: false,
        minCount: 0,
        maxCount: 2,
        minSize: [1.8, 0.8, 0.9],
        maxSize: [2.5, 1.0, 1.0],
        allowedPositions: 'any',
        spacing: { clearance: 0.3 },
      },
      {
        id: 'table',
        name: 'Coffee Table',
        required: false,
        minCount: 0,
        maxCount: 2,
        minSize: [0.8, 0.4, 0.8],
        maxSize: [1.5, 0.5, 1.2],
        allowedPositions: 'center',
        dependencies: ['sofa'],
      },
      {
        id: 'tv_stand',
        name: 'TV Stand',
        required: false,
        minCount: 0,
        maxCount: 1,
        minSize: [1.2, 0.5, 0.4],
        maxSize: [2.0, 0.7, 0.6],
        allowedPositions: 'wall',
      },
      {
        id: 'chair',
        name: 'Chair',
        required: false,
        minCount: 0,
        maxCount: 4,
        minSize: [0.6, 0.8, 0.6],
        maxSize: [0.8, 1.0, 0.8],
        allowedPositions: 'any',
      },
    ],
    constraints: {
      minObjects: 0,
      maxObjects: 10,
      layoutRules: [
        {
          id: 'sofa_tv_orientation',
          name: 'Sofa-TV Orientation',
          type: 'orientation',
          description: 'Sofa should face TV stand',
          categoryIds: ['sofa', 'tv_stand'],
          parameters: { facingAngle: 180 },
          priority: 'recommended',
        },
        {
          id: 'table_sofa_proximity',
          name: 'Table-Sofa Proximity',
          type: 'proximity',
          description: 'Coffee table should be near sofa',
          categoryIds: ['table', 'sofa'],
          parameters: { maxDistance: 0.5 },
          priority: 'recommended',
        },
      ],
    },
    zones: [
      {
        id: 'seating_zone',
        name: 'Seating Zone',
        bounds: { x: 1, y: 1, width: 3, height: 2 },
        preferredCategories: ['sofa', 'chair', 'table'],
      },
      {
        id: 'entertainment_zone',
        name: 'Entertainment Zone',
        bounds: { x: 0, y: 0, width: 5, height: 1 },
        preferredCategories: ['tv_stand'],
      },
    ],
  },
  // Placeholder configs for other room types
  closet: {
    roomType: 'closet',
    name: 'Closet',
    description: 'Closet layout',
    defaultDimensions: { width: 2, length: 1.5, height: 2.5 },
    categories: [],
    constraints: {},
  },
  dining_room: {
    roomType: 'dining_room',
    name: 'Dining Room',
    description: 'Dining room layout',
    defaultDimensions: { width: 4, length: 3, height: 2.5 },
    categories: [
      {
        id: 'table',
        name: 'Dining Table',
        required: true,
        minCount: 1,
        maxCount: 1,
        minSize: [1.2, 0.75, 0.8],
        maxSize: [2.0, 0.8, 1.5],
        allowedPositions: 'center',
        spacing: { clearance: 0.6 },
      },
      {
        id: 'chair',
        name: 'Dining Chair',
        required: false,
        minCount: 2,
        maxCount: 8,
        minSize: [0.4, 0.8, 0.4],
        maxSize: [0.5, 1.0, 0.5],
        allowedPositions: 'any',
        dependencies: ['table'],
      },
    ],
    constraints: {
      requiredCategories: ['table'],
      layoutRules: [
        {
          id: 'chair_table_proximity',
          name: 'Chair-Table Proximity',
          type: 'proximity',
          description: 'Chairs should be around table',
          categoryIds: ['chair', 'table'],
          parameters: { maxDistance: 0.4 },
          priority: 'required',
        },
      ],
    },
  },
  office: {
    roomType: 'office',
    name: 'Office',
    description: 'Office layout',
    defaultDimensions: { width: 3, length: 3, height: 2.5 },
    categories: [
      {
        id: 'desk',
        name: 'Desk',
        required: true,
        minCount: 1,
        maxCount: 2,
        minSize: [1.2, 0.75, 0.6],
        maxSize: [2.0, 0.8, 0.8],
        allowedPositions: 'wall',
      },
      {
        id: 'chair',
        name: 'Office Chair',
        required: true,
        minCount: 1,
        maxCount: 2,
        minSize: [0.5, 1.0, 0.5],
        maxSize: [0.6, 1.2, 0.6],
        allowedPositions: 'any',
        dependencies: ['desk'],
      },
    ],
    constraints: {
      requiredCategories: ['desk', 'chair'],
      layoutRules: [
        {
          id: 'chair_desk_proximity',
          name: 'Chair-Desk Proximity',
          type: 'proximity',
          description: 'Chair should be near desk',
          categoryIds: ['chair', 'desk'],
          parameters: { maxDistance: 0.3 },
          priority: 'required',
        },
      ],
    },
  },
  other: {
    roomType: 'other',
    name: 'Other',
    description: 'Generic room layout',
    defaultDimensions: { width: 4, length: 3, height: 2.5 },
    categories: [],
    constraints: {},
  },
};

/**
 * Get configuration for a room type
 */
export function getRoomTypeConfig(roomType: RoomType): RoomTypeConfig {
  return ROOM_TYPE_CONFIGS[roomType] || ROOM_TYPE_CONFIGS.other;
}

/**
 * Get categories for a room type
 */
export function getCategoriesForRoomType(roomType: RoomType): ObjectCategory[] {
  return getRoomTypeConfig(roomType).categories;
}

/**
 * Get required categories for a room type
 */
export function getRequiredCategoriesForRoomType(roomType: RoomType): string[] {
  const config = getRoomTypeConfig(roomType);
  return config.categories.filter(cat => cat.required).map(cat => cat.id);
}
