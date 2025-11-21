import { NextRequest, NextResponse } from 'next/server';
import {
  getTeamForUser,
  getRoomsForProject,
  createRoom,
  getUser,
  getProjectWithRooms,
} from '@/lib/db/queries';
import { ROOM_TYPES } from '@3dix/types';
import { z } from 'zod';
import {
  mapRoomsFromDb,
  mapRoomFromDb,
  mapClientRoomType,
} from '@/lib/db/roomTypeMapping';

const clientRoomTypeSchema = z.enum(ROOM_TYPES);

const createRoomSchema = z.object({
  name: z.string().min(1).max(255),
  roomType: clientRoomTypeSchema,
  width: z.number().positive().optional(),
  height: z.number().positive().optional(),
  length: z.number().positive().optional(),
});

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ projectId: string }> }
) {
  try {
    const { projectId } = await params;
    const projectIdNum = parseInt(projectId, 10);

    if (isNaN(projectIdNum)) {
      return NextResponse.json({ error: 'Invalid project ID' }, { status: 400 });
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

    const rooms = await getRoomsForProject(projectIdNum);
    return NextResponse.json(mapRoomsFromDb(rooms));
  } catch (error) {
    console.error('Error fetching rooms:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ projectId: string }> }
) {
  try {
    const { projectId } = await params;
    const projectIdNum = parseInt(projectId, 10);

    if (isNaN(projectIdNum)) {
      return NextResponse.json({ error: 'Invalid project ID' }, { status: 400 });
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
    const validatedData = createRoomSchema.parse(body);

    const room = await createRoom({
      projectId: projectIdNum,
      name: validatedData.name,
      roomType: mapClientRoomType(validatedData.roomType),
      width: validatedData.width || null,
      height: validatedData.height || null,
      length: validatedData.length || null,
      createdBy: user.id,
    });

    return NextResponse.json(mapRoomFromDb(room), { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      );
    }
    console.error('Error creating room:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
