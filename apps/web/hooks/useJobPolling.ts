'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { GenerationJobStatus } from '@3dix/types';

export interface JobStatus {
  id: number;
  status: GenerationJobStatus;
  progress: number | null;
  progressMessage: string | null;
  responseData: any;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

export function useJobPolling(jobId: number | null, pollInterval: number = 2000) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const isPollingRef = useRef(false);

  const fetchJob = useCallback(async () => {
    if (!jobId) return;

    try {
      setLoading(true);
      const response = await fetch(`/api/jobs/${jobId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch job');
      }
      const data = await response.json();
      setJob(data);
      setError(null);

      // Stop polling if job is completed or failed
      if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
        stopPolling();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      stopPolling();
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  const startPolling = useCallback(() => {
    if (!jobId || isPollingRef.current) return;

    isPollingRef.current = true;
    fetchJob(); // Initial fetch

    pollingRef.current = setInterval(() => {
      fetchJob();
    }, pollInterval);
  }, [jobId, pollInterval, fetchJob]);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    isPollingRef.current = false;
  }, []);

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
