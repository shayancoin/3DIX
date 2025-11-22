'use client';

import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

type AnalyticsResponse = {
  plan: { id: string; name: string; limits: { projects: number; rooms: number; jobsPerMonth: number; members: number } };
  usage: { projects: number; rooms: number; jobs: number };
  jobs: { total: number; completed: number; failed: number };
  limits: { canRunJob: boolean; code: string | null; message: string | null };
};

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/analytics')
      .then((res) => res.json())
      .then((json) => {
        if (json.error) {
          setError(json.error);
        } else {
          setData(json);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="p-6">
        <div className="text-destructive">Failed to load analytics: {error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="p-6">
        <div className="text-muted-foreground">Loading analytics...</div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Analytics</h1>
          <p className="text-muted-foreground text-sm">Team usage and job performance</p>
        </div>
        <Badge>{data.plan.name}</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard title="Projects" value={data.usage.projects} limit={data.plan.limits.projects} />
        <StatCard title="Rooms" value={data.usage.rooms} limit={data.plan.limits.rooms} />
        <StatCard title="Jobs (total)" value={data.jobs.total} limit={data.plan.limits.jobsPerMonth} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard title="Jobs Completed" value={data.jobs.completed} />
        <StatCard title="Jobs Failed" value={data.jobs.failed} />
        <Card className="p-4">
          <div className="text-sm text-muted-foreground">Job Capacity</div>
          <div className="text-lg font-semibold">
            {data.limits.canRunJob ? 'Within limit' : 'Limit reached'}
          </div>
          {!data.limits.canRunJob && (
            <div className="text-xs text-destructive mt-1">{data.limits.message}</div>
          )}
        </Card>
      </div>
    </div>
  );
}

function StatCard({ title, value, limit }: { title: string; value: number; limit?: number }) {
  return (
    <Card className="p-4">
      <div className="text-sm text-muted-foreground">{title}</div>
      <div className="text-2xl font-semibold">{value}</div>
      {typeof limit === 'number' && (
        <div className="text-xs text-muted-foreground mt-1">Limit: {limit}</div>
      )}
    </Card>
  );
}
