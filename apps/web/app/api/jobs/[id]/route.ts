import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import {
  getLayoutJob,
  getRoomWithProjectByRoomId,
  getTeamForUser,
  getUser,
  updateLayoutJob,
} from '@/lib/db/queries';
import { JobStatus } from '@/lib/jobs/stateMachine';

const updateJobSchema = z.object({
  status: z.enum(['queued', 'running', 'completed', 'failed', 'cancelled']).optional(),
  progress: z.number().min(0).max(100).optional(),
  progressMessage: z.string().optional(),
  responseData: z.any().optional(),
  errorMessage: z.string().optional(),
  errorDetails: z.any().optional(),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
});

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;
    const jobId = parseInt(id, 10);

    if (isNaN(jobId)) {
      return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
    }

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const job = await getLayoutJob(jobId);
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const room = await getRoomWithProjectByRoomId(job.roomId);
    if (!room || room.project.teamId !== team.id) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    return NextResponse.json(job);
  } catch (error) {
    console.error('Error fetching job:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;
    const jobId = parseInt(id, 10);

    if (isNaN(jobId)) {
      return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
    }

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const currentJob = await getLayoutJob(jobId);
    if (!currentJob) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const room = await getRoomWithProjectByRoomId(currentJob.roomId);
    if (!room || room.project.teamId !== team.id) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    const body = await req.json();
    const validatedData = updateJobSchema.parse(body);

    // Convert date strings to Date objects
    const updates: any = { ...validatedData };
    if (validatedData.startedAt) {
      updates.startedAt = new Date(validatedData.startedAt);
    }
    if (validatedData.completedAt) {
      updates.completedAt = new Date(validatedData.completedAt);
    }

    const job = await updateLayoutJob(jobId, updates as { status?: JobStatus });
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    return NextResponse.json(job);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      );
    }
    if (error instanceof Error && error.message.includes('Invalid job status transition')) {
      return NextResponse.json(
        { error: error.message },
        { status: 400 },
      );
    }
    console.error('Error updating job:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
