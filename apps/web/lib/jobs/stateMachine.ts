export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

/**
 * Finite state machine for layout jobs.
 *
 * S = { queued, running, completed, failed, cancelled }
 * Allowed transitions:
 *   queued  -> running | failed | cancelled
 *   running -> completed | failed | cancelled
 * Terminal states (completed, failed, cancelled) cannot transition further.
 */
const allowedTransitions: Record<JobStatus, JobStatus[]> = {
  queued: ['running', 'failed', 'cancelled'],
  running: ['completed', 'failed', 'cancelled'],
  completed: [],
  failed: [],
  cancelled: [],
};

export function isValidTransition(prev: JobStatus, next: JobStatus): boolean {
  if (prev === next) return true;
  return allowedTransitions[prev]?.includes(next) ?? false;
}

export function assertValidTransition(prev: JobStatus, next: JobStatus): void {
  if (!isValidTransition(prev, next)) {
    throw new Error(`Invalid job status transition: ${prev} -> ${next}`);
  }
}

export const TERMINAL_STATUSES: JobStatus[] = ['completed', 'failed', 'cancelled'];
