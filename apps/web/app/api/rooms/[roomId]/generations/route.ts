import { NextRequest, NextResponse } from 'next/server';
import { getTeamForUser, getRoomGenerations, createRoomGeneration, getRoomById, getProjectWithRooms } from '@/lib/db/queries';
import { getUser } from '@/lib/db/queries';
import { z } from 'zod';

const createGenerationSchema = z.object({
  vibeSpec: z.any(), // VibeSpec type
  status: z.enum(['stub', 'generated', 'completed']).optional().default('stub'),
});

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ roomId: string }> }
) {
  try {
    const { roomId } = await params;
    const roomIdNum = parseInt(roomId, 10);

    if (isNaN(roomIdNum)) {
      return NextResponse.json({ error: 'Invalid room ID' }, { status: 400 });
    }

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    // Verify room belongs to team (via project)
    const room = await getRoomById(roomIdNum);
    if (!room) {
      return NextResponse.json({ error: 'Room not found' }, { status: 404 });
    }

    // Verify project belongs to team
    const project = await getProjectWithRooms(room.projectId, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Room not found' }, { status: 404 });
    }

    const generations = await getRoomGenerations(roomIdNum);
    return NextResponse.json(generations);
  } catch (error) {
    console.error('Error fetching generations:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ roomId: string }> }
) {
  try {
    const { roomId } = await params;
    const roomIdNum = parseInt(roomId, 10);

    if (isNaN(roomIdNum)) {
      return NextResponse.json({ error: 'Invalid room ID' }, { status: 400 });
    }

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const body = await req.json();
    const validatedData = createGenerationSchema.parse(body);

    const generation = await createRoomGeneration({
      roomId: roomIdNum,
      vibeSpec: validatedData.vibeSpec,
      status: validatedData.status || 'stub',
    });

    return NextResponse.json(generation, { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      );
    }
    console.error('Error creating generation:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
