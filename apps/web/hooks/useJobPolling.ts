'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { GenerationJobStatus, LayoutJobStatusResponse } from '@3dix/types';

export interface JobStatus {
  id: number;
  status: GenerationJobStatus;
  progress: number | null;
  progressMessage: string | null;
  responseData: any;
  result?: any;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

const normalizeJob = (data: LayoutJobStatusResponse): JobStatus => {
  const progress = data.progress ?? 0;
  return {
    id: typeof data.job_id === 'string' ? parseInt(data.job_id, 10) : data.job_id,
    status: data.status,
    progress: Number.isFinite(progress) ? progress : 0,
    progressMessage: data.progress_message ?? null,
    responseData: data.result ?? null,
    result: data.result ?? null,
    errorMessage: data.error ?? null,
    createdAt: data.created_at,
    updatedAt:
      data.updated_at ||
      data.completed_at ||
      data.started_at ||
      data.created_at,
  };
};

/**
 * Continuously polls the layout job status endpoint for a given job ID and exposes the latest normalized status and polling controls.
 *
 * @param jobId - The layout job identifier to poll; if `null` polling is disabled.
 * @param pollInterval - Time in milliseconds between poll requests (default: 2000).
 * @returns An object containing:
 *   - `job`: the latest normalized JobStatus or `null`
 *   - `loading`: `true` while a fetch is in progress, `false` otherwise
 *   - `error`: an error message string or `null`
 *   - `refetch`: a function to fetch the latest job status immediately
 *   - `startPolling`: a function to start polling (no-op if already polling or `jobId` is falsy)
 *   - `stopPolling`: a function to stop polling
 */
export function useJobPolling(jobId: number | string | null, pollInterval: number = 2000) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const isPollingRef = useRef(false);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    isPollingRef.current = false;
  }, []);

  const fetchJob = useCallback(async () => {
    if (!jobId) return;

    try {
      setLoading(true);
      const response = await fetch(`/api/layout-jobs/${jobId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch job');
      }
      const data: LayoutJobStatusResponse = await response.json();
      const normalized = normalizeJob(data);
      setJob(normalized);
      setError(null);

      // Stop polling if job is completed or failed
      if (
        normalized.status === 'completed' ||
        normalized.status === 'failed' ||
        normalized.status === 'cancelled'
      ) {
        stopPolling();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      stopPolling();
    } finally {
      setLoading(false);
    }
  }, [jobId, stopPolling]);

  const startPolling = useCallback(() => {
    if (!jobId || isPollingRef.current) return;

    isPollingRef.current = true;
    fetchJob(); // Initial fetch

    pollingRef.current = setInterval(() => {
      fetchJob();
    }, pollInterval);
  }, [jobId, pollInterval, fetchJob]);

  useEffect(() => {
    if (jobId) {
      startPolling();
    } else {
      stopPolling();
      setJob(null);
    }

    return () => {
      stopPolling();
    };
  }, [jobId, startPolling, stopPolling]);

  return {
    job,
    loading,
    error,
    refetch: fetchJob,
    startPolling,
    stopPolling,
  };
}