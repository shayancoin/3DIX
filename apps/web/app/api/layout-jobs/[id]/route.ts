import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import {
  getLayoutJob,
  getRoomWithProjectByRoomId,
  getTeamForUser,
  getUser,
} from '@/lib/db/queries';
import { LayoutJobStatusResponse } from '@3dix/types';

const paramsSchema = z.object({
  id: z.string(),
});

const serializeJob = (job: any): LayoutJobStatusResponse => ({
  job_id: job.id,
  status: job.status,
  result: job.responseData ?? null,
  error: job.errorMessage ?? null,
  progress: job.progress ?? 0,
  progress_message: job.progressMessage ?? null,
  created_at: job.createdAt ? new Date(job.createdAt).toISOString() : new Date().toISOString(),
  started_at: job.startedAt ? new Date(job.startedAt).toISOString() : null,
  completed_at: job.completedAt ? new Date(job.completedAt).toISOString() : null,
  updated_at: job.updatedAt ? new Date(job.updatedAt).toISOString() : null,
});

/**
 * Fetches and returns the status of a layout job identified by the route `id`.
 *
 * @param req - The incoming Next.js request
 * @param params - Route parameters; `params.id` is the job id string
 * @returns A NextResponse containing the serialized job status on success; on error, a JSON payload with an `error` object and one of the codes `BAD_REQUEST`, `UNAUTHORIZED`, `NO_TEAM`, `NOT_FOUND`, `FORBIDDEN`, or `INTERNAL_ERROR`.
 */
export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = paramsSchema.parse(params);
    const jobId = parseInt(id, 10);

    if (Number.isNaN(jobId)) {
      return NextResponse.json({ error: { code: 'BAD_REQUEST', message: 'Invalid job ID' } }, { status: 400 });
    }

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: { code: 'UNAUTHORIZED', message: 'Unauthorized' } }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: { code: 'NO_TEAM', message: 'No team found' } }, { status: 404 });
    }

    const job = await getLayoutJob(jobId);
    if (!job) {
      return NextResponse.json({ error: { code: 'NOT_FOUND', message: 'Job not found' } }, { status: 404 });
    }

    const room = await getRoomWithProjectByRoomId(job.roomId);
    if (!room || room.project.teamId !== team.id) {
      return NextResponse.json({ error: { code: 'FORBIDDEN', message: 'Forbidden' } }, { status: 403 });
    }

    return NextResponse.json(serializeJob(job));
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: { code: 'BAD_REQUEST', message: 'Invalid params', details: error.errors } },
        { status: 400 },
      );
    }
    console.error('Error fetching layout job:', error);
    return NextResponse.json(
      { error: { code: 'INTERNAL_ERROR', message: 'Internal server error' } },
      { status: 500 },
    );
  }
}