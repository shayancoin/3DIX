import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import {
  createLayoutJob,
  getLayoutJobsForRoom,
  getRoomWithProjectByRoomId,
  getTeamForUser,
  getUser,
} from '@/lib/db/queries';
import { JobStatus } from '@/lib/jobs/stateMachine';
import { VibeSpec } from '@3dix/types';

const createJobSchema = z.object({
  roomId: z.union([z.number(), z.string()]),
  requestData: z.any().optional(), // LayoutRequest
  vibeSpec: z.any().optional(),
  maskType: z.enum(['none', 'floor', 'arch']).optional(),
  archMaskUrl: z.string().url().optional(),
});

export async function POST(req: NextRequest) {
  try {
    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const body = await req.json();
    const validatedData = createJobSchema.parse(body);
    const roomIdNum =
      typeof validatedData.roomId === 'string'
        ? parseInt(validatedData.roomId, 10)
        : validatedData.roomId;

    if (Number.isNaN(roomIdNum)) {
      return NextResponse.json({ error: 'Invalid room ID' }, { status: 400 });
    }

    // Verify room belongs to user's team
    const room = await getRoomWithProjectByRoomId(roomIdNum);
    if (!room) {
      return NextResponse.json({ error: 'Room not found' }, { status: 404 });
    }

    if (room.project.teamId !== team.id) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    const requestData =
      validatedData.requestData ??
      {
        room_type: (room as any).roomType || 'living_room',
        arch_mask_url: validatedData.archMaskUrl ?? validatedData.requestData?.arch_mask_url,
        mask_type: validatedData.maskType ?? validatedData.requestData?.mask_type ?? 'none',
        vibe_spec: (validatedData.vibeSpec || {}) as VibeSpec,
        seed: validatedData.requestData?.seed,
      };

    const job = await createLayoutJob({
      roomId: roomIdNum,
      status: 'queued' as JobStatus,
      requestData,
      progress: 0,
    });

    return NextResponse.json(job, { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      );
    }
    console.error('Error creating job:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET(req: NextRequest) {
  try {
    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const roomId = searchParams.get('roomId');

    if (roomId) {
      const roomIdNum = parseInt(roomId, 10);
      if (isNaN(roomIdNum)) {
        return NextResponse.json({ error: 'Invalid room ID' }, { status: 400 });
      }

      const room = await getRoomWithProjectByRoomId(roomIdNum);
      if (!room) {
        return NextResponse.json({ error: 'Room not found' }, { status: 404 });
      }

      if (room.project.teamId !== team.id) {
        return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
      }

      const jobs = await getLayoutJobsForRoom(roomIdNum);
      return NextResponse.json(jobs);
    }

    return NextResponse.json({ error: 'roomId parameter required' }, { status: 400 });
  } catch (error) {
    console.error('Error fetching jobs:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
