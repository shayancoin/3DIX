import crypto from 'node:crypto';
import { getQueuedJobs, updateLayoutJob } from '../lib/db/queries';
import { JobStatus } from '../lib/jobs/stateMachine';
import { LayoutRequest, LayoutResponse, RoomType } from '@3dix/types';

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const buildStubLayoutResponse = (seed: number): LayoutResponse => ({
  semantic_map_png_url: 'https://placehold.co/600x400.png',
  world_scale: 0.01,
  objects: [
    {
      id: 'sofa-1',
      category: 'sofa',
      position: [1, 0, 2],
      size: [2, 1, 1],
      orientation: 1,
      metadata: { seed },
    },
    {
      id: 'table-1',
      category: 'table',
      position: [0, 0, 0],
      size: [1, 0.8, 1],
      orientation: 0,
      metadata: { seed },
    },
  ],
});

const normalizeRequest = (data: any): LayoutRequest => {
  const roomType = (data?.room_type as RoomType) || 'living_room';
  return {
    room_type: roomType,
    arch_mask_url: data?.arch_mask_url,
    mask_type: data?.mask_type || 'none',
    seed: data?.seed ?? crypto.randomInt(1, 10_000),
    vibe_spec: data?.vibe_spec ?? {
      prompt: {
        text: data?.prompt ?? 'A cozy room',
        roomType,
      },
      tags: [],
      sliders: [],
    },
  };
};

/**
 * Sends a layout generation request to the configured layout ML service and returns the generated layout response.
 *
 * @param request - The layout generation request payload to send to the ML service.
 * @returns The parsed layout generation response.
 * @throws Error if the ML service responds with a non-success HTTP status.
 */
async function callLayoutService(request: LayoutRequest): Promise<LayoutResponse> {
  const mlUrl = process.env.LAYOUT_ML_URL || 'http://localhost:8001/generate-layout';
  const res = await fetch(mlUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    throw new Error(`ML service error: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as LayoutResponse;
}

/**
 * Processes a layout generation job: marks it running, invokes the ML layout service (falls back to a deterministic stub on failure), and updates the job to completed or failed.
 *
 * Normalizes the job's request data before calling the ML service, measures processing time, and merges processing metadata (`processingTimeMs` and `generator: 'layout-worker'`) into the response stored on success. On any error the job is marked failed and the error message is recorded.
 *
 * @param job - Job object containing at minimum `id` and `requestData`; `id` is used to update job status and `requestData` is normalized and sent to the layout service
 */
async function processJob(job: any) {
  const start = Date.now();

  await updateLayoutJob(job.id, {
    status: 'running' as JobStatus,
    progress: 10,
    progressMessage: 'Starting layout generation...',
    startedAt: new Date(),
  });

  try {
    const mlRequest = normalizeRequest(job.requestData);
    let responseData: LayoutResponse;
    try {
      responseData = await callLayoutService(mlRequest);
    } catch (err) {
      console.warn(`ML service not available, falling back to stub: ${String(err)}`);
      responseData = buildStubLayoutResponse(mlRequest.seed ?? crypto.randomInt(1, 10_000));
    }

    const durationMs = Date.now() - start;
    await updateLayoutJob(job.id, {
      status: 'completed' as JobStatus,
      responseData: {
        ...responseData,
        metadata: {
          ...(responseData as any).metadata,
          processingTimeMs: durationMs,
          generator: 'layout-worker',
        },
      },
      progress: 100,
      progressMessage: 'Layout generation completed.',
      completedAt: new Date(),
    });
  } catch (error: any) {
    await updateLayoutJob(job.id, {
      status: 'failed' as JobStatus,
      progress: 100,
      errorMessage: error?.message ?? 'Unknown worker error',
      progressMessage: 'Layout generation failed.',
      completedAt: new Date(),
    });
  }
}

/**
 * Fetches up to five queued layout jobs and processes each sequentially.
 *
 * This polls the job queue for up to 5 jobs and processes them one by one.
 */
export async function pollOnce() {
  const queued = await getQueuedJobs(5);
  for (const job of queued) {
    await processJob(job);
  }
}

/**
 * Continuously polls for queued layout jobs and processes them on a fixed interval.
 *
 * Reads the polling interval from the `LAYOUT_WORKER_POLL_INTERVAL_MS` environment variable (defaults to 3000 ms),
 * then repeatedly calls `pollOnce()` followed by a sleep for the configured interval.
 */
export async function runWorkerLoop() {
  const pollIntervalMs = parseInt(process.env.LAYOUT_WORKER_POLL_INTERVAL_MS ?? '3000', 10);
  // eslint-disable-next-line no-constant-condition
  while (true) {
    await pollOnce();
    await sleep(pollIntervalMs);
  }
}

// Execute when run directly via ts-node / node
runWorkerLoop().catch((err) => {
  console.error('Worker terminated with error:', err);
  process.exit(1);
});