import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import {
  createLayoutJob,
  getRoomWithProjectByRoomId,
  getTeamForUser,
  getUser,
} from '@/lib/db/queries';
import { JobStatus } from '@/lib/jobs/stateMachine';
import { VibeSpec, ROOM_TYPE_CONFIGS, RoomTypeConfig } from '@3dix/types';

const toVibeSpec = (vibe: any, roomType: string): VibeSpec => {
  return {
    prompt: {
      text: vibe?.prompt ?? '',
      roomType: roomType as any,
    },
    tags: Array.isArray(vibe?.keywords)
      ? vibe.keywords.map((k: string, idx: number) => ({
          id: `${idx}-${k}`,
          label: k,
          category: 'style',
          weight: 0.5,
        }))
      : [],
    sliders: Object.entries(vibe?.style_sliders ?? {}).map(([key, value]) => ({
      id: key,
      label: key,
      min: 0,
      max: 1,
      value: typeof value === 'number' ? value : 0.5,
    })),
    metadata: {
      createdAt: new Date().toISOString(),
    },
  };
};

const createLayoutJobSchema = z.object({
  room_id: z.union([z.number(), z.string()]),
  vibe_spec: z.any(),
  constraints: z
    .object({
      arch_mask_url: z.string().url().optional(),
      mask_type: z.enum(['none', 'floor', 'arch']).optional(),
      seed: z.number().optional(),
    })
    .optional(),
});

/**
 * Create a layout generation job for a room and return the job's id and status.
 *
 * Validates the request body, verifies user and team membership, ensures the room
 * belongs to the team, transforms the provided vibe spec, and enqueues a job.
 *
 * @param req - Incoming request whose JSON body must include `room_id`, `vibe_spec`, and optional `constraints` (`arch_mask_url`, `mask_type`, `seed`)
 * @returns The created job's identifier and current status: `{ job_id: number, status: string }`. On failure the response body contains an `error` object with `code` and `message` (and `details` for validation errors).
 */
export async function POST(req: NextRequest) {
  try {
    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: { code: 'UNAUTHORIZED', message: 'Unauthorized' } }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: { code: 'NO_TEAM', message: 'No team found' } }, { status: 404 });
    }

    const body = await req.json();
    const validated = createLayoutJobSchema.parse(body);
    const roomId = typeof validated.room_id === 'string' ? parseInt(validated.room_id, 10) : validated.room_id;

    if (Number.isNaN(roomId)) {
      return NextResponse.json({ error: { code: 'BAD_REQUEST', message: 'Invalid room_id' } }, { status: 400 });
    }

    const room = await getRoomWithProjectByRoomId(roomId);
    if (!room) {
      return NextResponse.json({ error: { code: 'NOT_FOUND', message: 'Room not found' } }, { status: 404 });
    }

    if (room.project.teamId !== team.id) {
      return NextResponse.json({ error: { code: 'FORBIDDEN', message: 'Room does not belong to your team' } }, { status: 403 });
    }

    const roomType = (room as any).roomType || (room as any).type || 'living_room';
    const baseConfig = ROOM_TYPE_CONFIGS[roomType as keyof typeof ROOM_TYPE_CONFIGS] ?? ROOM_TYPE_CONFIGS.other;
    const roomConfig: RoomTypeConfig = {
      ...baseConfig,
      defaultDimensions: {
        ...baseConfig.defaultDimensions,
        width: room.width ?? baseConfig.defaultDimensions.width,
        length: room.length ?? baseConfig.defaultDimensions.length,
        height: room.height ?? baseConfig.defaultDimensions.height,
      },
      categories: (baseConfig.categories ?? []).map((cat) => ({
        ...cat,
        spacing: cat.spacing ? { ...cat.spacing } : undefined,
        dependencies: cat.dependencies ? [...cat.dependencies] : undefined,
        conflicts: cat.conflicts ? [...cat.conflicts] : undefined,
      })),
      constraints: { ...(baseConfig.constraints ?? {}) },
      zones: baseConfig.zones ? [...baseConfig.zones] : undefined,
    };

    const requestData = {
      room_type: roomType,
      room_config: roomConfig,
      arch_mask_url: validated.constraints?.arch_mask_url,
      mask_type: validated.constraints?.mask_type ?? 'none',
      vibe_spec: toVibeSpec(validated.vibe_spec, roomType),
      seed: validated.constraints?.seed,
    };

    const job = await createLayoutJob({
      roomId,
      requestData,
      status: 'queued' as JobStatus,
      progress: 0,
      progressMessage: 'Queued',
    });

    return NextResponse.json({ job_id: job.id, status: job.status });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: { code: 'BAD_REQUEST', message: 'Invalid request data', details: error.errors } },
        { status: 400 },
      );
    }
    console.error('Error creating layout job:', error);
    return NextResponse.json(
      { error: { code: 'INTERNAL_ERROR', message: 'Internal server error' } },
      { status: 500 },
    );
  }
}

/**
 * Indicates that listing layout jobs via this endpoint is not implemented and directs clients to the correct endpoint.
 *
 * @returns A JSON response containing an `error` object with `code` set to `"NOT_IMPLEMENTED"` and `message` set to `"List jobs via /api/jobs?roomId=<id>"`, returned with HTTP status 400.
 */
export async function GET(req: NextRequest) {
  return NextResponse.json({ error: { code: 'NOT_IMPLEMENTED', message: 'List jobs via /api/jobs?roomId=<id>' } }, { status: 400 });
}
