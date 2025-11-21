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

export async function pollOnce() {
  const queued = await getQueuedJobs(5);
  for (const job of queued) {
    await processJob(job);
  }
}

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
