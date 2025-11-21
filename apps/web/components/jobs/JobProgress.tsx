'use client';

import React from 'react';
import { JobStatus } from '@/hooks/useJobPolling';
import { CheckCircle2, XCircle, Loader2, Clock } from 'lucide-react';
import { Card } from '@/components/ui/card';

interface JobProgressProps {
  job: JobStatus | null;
  loading?: boolean;
}

export function JobProgress({ job, loading }: JobProgressProps) {
  if (!job && !loading) {
    return null;
  }

  if (loading && !job) {
    return (
      <Card className="p-4">
        <div className="flex items-center gap-3">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <div>
            <div className="font-medium">Loading job status...</div>
          </div>
        </div>
      </Card>
    );
  }

  if (!job) {
    return null;
  }

  const getStatusIcon = () => {
    switch (job.status) {
      case 'completed':
        return <CheckCircle2 className="h-5 w-5 text-green-600" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-600" />;
      case 'running':
        return <Loader2 className="h-5 w-5 animate-spin text-primary" />;
      case 'queued':
        return <Clock className="h-5 w-5 text-yellow-600" />;
      default:
        return <Clock className="h-5 w-5 text-gray-600" />;
    }
  };

  const getStatusColor = () => {
    switch (job.status) {
      case 'completed':
        return 'text-green-600';
      case 'failed':
        return 'text-red-600';
      case 'running':
        return 'text-primary';
      case 'queued':
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
    }
  };

  const progress = job.progress ?? 0;

  return (
    <Card className="p-4">
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          {getStatusIcon()}
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <div className={`font-medium capitalize ${getStatusColor()}`}>
                {job.status === 'queued' && 'Queued'}
                {job.status === 'running' && 'Processing'}
                {job.status === 'completed' && 'Completed'}
                {job.status === 'failed' && 'Failed'}
                {job.status === 'cancelled' && 'Cancelled'}
              </div>
              {job.progress !== null && (
                <div className="text-sm text-muted-foreground">{progress}%</div>
              )}
            </div>
            {job.progressMessage && (
              <div className="text-sm text-muted-foreground mt-1">
                {job.progressMessage}
              </div>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        {(job.status === 'running' || job.status === 'queued') && (
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-primary h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {/* Error Message */}
        {job.status === 'failed' && job.errorMessage && (
          <div className="text-sm text-red-600 bg-red-50 p-2 rounded">
            {job.errorMessage}
          </div>
        )}

        {/* Success Message */}
        {job.status === 'completed' && (
          <div className="text-sm text-green-600 bg-green-50 p-2 rounded">
            Layout generation completed successfully!
          </div>
        )}

        {/* Timestamps */}
        <div className="text-xs text-muted-foreground">
          Created: {new Date(job.createdAt).toLocaleString()}
          {job.updatedAt !== job.createdAt && (
            <> • Updated: {new Date(job.updatedAt).toLocaleString()}</>
          )}
        </div>
      </div>
    </Card>
  );
}
