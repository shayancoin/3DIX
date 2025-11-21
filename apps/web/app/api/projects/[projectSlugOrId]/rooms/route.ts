import { NextRequest, NextResponse } from 'next/server';
import { getTeamForUser, getRoomsForProject, createRoom, getProjectByIdentifier } from '@/lib/db/queries';
import { getUser } from '@/lib/db/queries';
import { RoomType } from '@/lib/db/schema';
import { z } from 'zod';

const createRoomSchema = z.object({
  name: z.string().min(1).max(255),
  roomType: z.nativeEnum(RoomType),
  width: z.number().positive().optional(),
  height: z.number().positive().optional(),
  length: z.number().positive().optional(),
  floorplanUrl: z.string().url().optional(),
});

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ projectSlugOrId: string }> }
) {
  try {
    const { projectSlugOrId } = await params;

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    // Verify project belongs to team
    const project = await getProjectByIdentifier(projectSlugOrId, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const rooms = await getRoomsForProject(project.id);
    return NextResponse.json(rooms);
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
  { params }: { params: Promise<{ projectSlugOrId: string }> }
) {
  try {
    const { projectSlugOrId } = await params;

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    // Verify project belongs to team
    const project = await getProjectByIdentifier(projectSlugOrId, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const body = await req.json();
    const validatedData = createRoomSchema.parse(body);

    const room = await createRoom({
      projectId: project.id,
      name: validatedData.name,
      roomType: validatedData.roomType,
      width: validatedData.width || null,
      height: validatedData.height || null,
      length: validatedData.length || null,
      floorplanUrl: validatedData.floorplanUrl || null,
      createdBy: user.id,
    });

    return NextResponse.json(room, { status: 201 });
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
