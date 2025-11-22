import { NextRequest, NextResponse } from 'next/server';
import { getRoomsForProject, createRoom, getProjectWithRooms, getTeamWithMembershipForUser } from '@/lib/db/queries';
import { RoomType } from '@/lib/db/schema';
import { hasRequiredRole } from '@/lib/auth/roles';
import { ensureRoomCapacity } from '@/lib/billing/limits';
import { z } from 'zod';

const createRoomSchema = z.object({
  name: z.string().min(1).max(255),
  roomType: z.nativeEnum(RoomType),
  width: z.number().positive().optional(),
  height: z.number().positive().optional(),
  length: z.number().positive().optional(),
});

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const projectIdNum = parseInt(id, 10);

    if (isNaN(projectIdNum)) {
      return NextResponse.json({ error: 'Invalid project ID' }, { status: 400 });
    }

    const ctx = await getTeamWithMembershipForUser();
    if (!ctx) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    const team = ctx.team;

    // Verify project belongs to team
    const project = await getProjectWithRooms(projectIdNum, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const rooms = await getRoomsForProject(projectIdNum);
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
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const projectIdNum = parseInt(id, 10);

    if (isNaN(projectIdNum)) {
      return NextResponse.json({ error: 'Invalid project ID' }, { status: 400 });
    }

    const ctx = await getTeamWithMembershipForUser();
    if (!ctx) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    if (!hasRequiredRole(ctx.role, 'member')) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }
    const { team, user } = ctx;

    // Verify project belongs to team
    const project = await getProjectWithRooms(projectIdNum, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const body = await req.json();
    const validatedData = createRoomSchema.parse(body);

    const capacity = await ensureRoomCapacity(team);
    if (!capacity.ok) {
      return NextResponse.json({ error: capacity.message, code: capacity.code }, { status: 402 });
    }

    const room = await createRoom({
      projectId: projectIdNum,
      name: validatedData.name,
      roomType: validatedData.roomType,
      width: validatedData.width || null,
      height: validatedData.height || null,
      length: validatedData.length || null,
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
