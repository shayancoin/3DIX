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

/**
 * Retrieve a layout job by ID if the requesting user belongs to the job's team.
 *
 * Validates the `id` path parameter, verifies the authenticated user and their team,
 * ensures the job exists and belongs to the team's project, and returns the job data.
 *
 * @returns The job object as JSON on success; an error JSON with one of these HTTP status codes on failure:
 * - 400: Invalid job ID or invalid request data
 * - 401: Unauthorized (no current user)
 * - 403: Forbidden (job does not belong to the user's team)
 * - 404: No team found or Job not found
 * - 500: Internal server error
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
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

/**
 * Updates fields of a layout job identified by ID and returns the updated job.
 *
 * @param params.id - The job ID path parameter as a string
 * @returns The updated job object as JSON on success; on failure, a JSON error object with an appropriate HTTP status code
 */
export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
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
