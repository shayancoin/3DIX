import { NextRequest, NextResponse } from 'next/server';
import { getTeamForUser } from '@/lib/db/queries';
import { getUser } from '@/lib/db/queries';
import { createLayoutJob, getLayoutJobsForRoom } from '@/lib/db/queries';
import { getRoomById, getProjectWithRooms } from '@/lib/db/queries';
import { z } from 'zod';

const createJobSchema = z.object({
  roomId: z.number(),
  requestData: z.any(), // LayoutRequest
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

    // Verify room belongs to user's team
    // We need to get the project first to verify team ownership
    const room = await getRoomById(validatedData.roomId);
    if (!room) {
      return NextResponse.json({ error: 'Room not found' }, { status: 404 });
    }

    // Verify project belongs to team
    const project = await getProjectWithRooms(room.projectId, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Room not found' }, { status: 404 });
    }

    const job = await createLayoutJob({
      roomId: validatedData.roomId,
      status: 'queued',
      requestData: validatedData.requestData,
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
