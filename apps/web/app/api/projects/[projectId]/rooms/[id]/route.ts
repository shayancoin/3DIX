import { NextRequest, NextResponse } from 'next/server';
import {
  getTeamForUser,
  getRoom,
  updateRoom,
  deleteRoom,
  getUser,
  getProjectWithRooms,
} from '@/lib/db/queries';
import { ROOM_TYPES } from '@3dix/types';
import { z } from 'zod';
import { mapRoomFromDb, mapClientRoomType } from '@/lib/db/roomTypeMapping';

const clientRoomTypeSchema = z.enum(ROOM_TYPES);

const updateRoomSchema = z.object({
  name: z.string().min(1).max(255).optional(),
  roomType: clientRoomTypeSchema.optional(),
  width: z.number().positive().optional().nullable(),
  height: z.number().positive().optional().nullable(),
  length: z.number().positive().optional().nullable(),
  layoutData: z.any().optional().nullable(),
  sceneData: z.any().optional().nullable(),
  vibeSpec: z.any().optional().nullable(),
  thumbnailUrl: z.string().url().optional().nullable(),
  isActive: z.boolean().optional(),
});

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ projectId: string; id: string }> }
) {
  try {
    const { projectId, id } = await params;
    const projectIdNum = parseInt(projectId, 10);
    const roomId = parseInt(id, 10);

    if (isNaN(projectIdNum) || isNaN(roomId)) {
      return NextResponse.json({ error: 'Invalid ID' }, { status: 400 });
    }

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    // Verify project belongs to team
    const project = await getProjectWithRooms(projectIdNum, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const room = await getRoom(roomId, projectIdNum);
    if (!room) {
      return NextResponse.json({ error: 'Room not found' }, { status: 404 });
    }

    return NextResponse.json(mapRoomFromDb(room));
  } catch (error) {
    console.error('Error fetching room:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ projectId: string; id: string }> }
) {
  try {
    const { projectId, id } = await params;
    const projectIdNum = parseInt(projectId, 10);
    const roomId = parseInt(id, 10);

    if (isNaN(projectIdNum) || isNaN(roomId)) {
      return NextResponse.json({ error: 'Invalid ID' }, { status: 400 });
    }

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    // Verify project belongs to team
    const project = await getProjectWithRooms(projectIdNum, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const body = await req.json();
    const validatedData = updateRoomSchema.parse(body);

    const updatePayload = {
      ...validatedData,
      roomType: validatedData.roomType
        ? mapClientRoomType(validatedData.roomType)
        : undefined,
    };

    const room = await updateRoom(roomId, projectIdNum, updatePayload);
    if (!room) {
      return NextResponse.json({ error: 'Room not found' }, { status: 404 });
    }

    return NextResponse.json(mapRoomFromDb(room));
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      );
    }
    console.error('Error updating room:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ projectId: string; id: string }> }
) {
  try {
    const { projectId, id } = await params;
    const projectIdNum = parseInt(projectId, 10);
    const roomId = parseInt(id, 10);

    if (isNaN(projectIdNum) || isNaN(roomId)) {
      return NextResponse.json({ error: 'Invalid ID' }, { status: 400 });
    }

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    // Verify project belongs to team
    const project = await getProjectWithRooms(projectIdNum, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    await deleteRoom(roomId, projectIdNum);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting room:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
