import { NextRequest, NextResponse } from 'next/server';
import { getTeamForUser, getRoomGeneration, updateRoomGeneration } from '@/lib/db/queries';
import { getUser } from '@/lib/db/queries';
import { getProjectWithRooms } from '@/lib/db/queries';
import { db } from '@/lib/db/drizzle';
import { eq } from 'drizzle-orm';
import { roomGenerations } from '@/lib/db/schema';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ roomId: string; generationId: string }> }
) {
  try {
    const { roomId, generationId } = await params;
    const roomIdNum = parseInt(roomId, 10);
    const generationIdNum = parseInt(generationId, 10);

    if (isNaN(roomIdNum) || isNaN(generationIdNum)) {
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

    const generation = await getRoomGeneration(generationIdNum);
    if (!generation || generation.roomId !== roomIdNum) {
      return NextResponse.json({ error: 'Generation not found' }, { status: 404 });
    }

    return NextResponse.json(generation);
  } catch (error) {
    console.error('Error fetching generation:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ roomId: string; generationId: string }> }
) {
  try {
    const { roomId, generationId } = await params;
    const roomIdNum = parseInt(roomId, 10);
    const generationIdNum = parseInt(generationId, 10);

    if (isNaN(roomIdNum) || isNaN(generationIdNum)) {
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

    // Verify generation belongs to room
    const generation = await getRoomGeneration(generationIdNum);
    if (!generation || generation.roomId !== roomIdNum) {
      return NextResponse.json({ error: 'Generation not found' }, { status: 404 });
    }

    await db.delete(roomGenerations).where(eq(roomGenerations.id, generationIdNum));

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting generation:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
