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

/**
 * Check whether a transition between two job statuses is allowed.
 *
 * Identical `prev` and `next` statuses are considered valid; non-identical
 * transitions are validated against the configured allowed transitions.
 *
 * @returns `true` if transitioning from `prev` to `next` is allowed, `false` otherwise.
 */
export function isValidTransition(prev: JobStatus, next: JobStatus): boolean {
  if (prev === next) return true;
  return allowedTransitions[prev]?.includes(next) ?? false;
}

/**
 * Ensures the transition from `prev` to `next` is allowed for a job status.
 *
 * @param prev - Current job status
 * @param next - Desired job status
 * @throws Error if the transition is not allowed (message: `Invalid job status transition: <prev> -> <next>`)
 */
export function assertValidTransition(prev: JobStatus, next: JobStatus): void {
  if (!isValidTransition(prev, next)) {
    throw new Error(`Invalid job status transition: ${prev} -> ${next}`);
  }
}

export const TERMINAL_STATUSES: JobStatus[] = ['completed', 'failed', 'cancelled'];