'use client';

import { ConstraintValidation } from '@3dix/types';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

type ConstraintStatusProps = {
  validation?: ConstraintValidation | null;
  loading?: boolean;
};

const severityBadge = (severity: string) =>
  cn(
    'px-2 py-0.5 rounded-full text-xs font-medium capitalize',
    severity === 'error' && 'bg-red-100 text-red-700',
    severity === 'warning' && 'bg-amber-100 text-amber-700',
    severity === 'info' && 'bg-blue-100 text-blue-700',
  );

export function ConstraintStatus({ validation, loading }: ConstraintStatusProps) {
  const formatNumber = (value?: number | null) => (typeof value === 'number' ? value.toFixed(2) : 'n/a');

  if (loading && !validation) {
    return (
      <Card className="p-4 flex items-center gap-2">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        <div className="text-sm text-muted-foreground">Checking constraints...</div>
      </Card>
    );
  }

  if (!validation) {
    return (
      <Card className="p-4">
        <div className="text-sm text-muted-foreground">No constraint data yet.</div>
      </Card>
    );
  }

  const violations = (validation.violations || [])
    .filter((v) => (v.normalized_violation ?? 0) > 0 || !validation.satisfied)
    .sort((a, b) => (b.normalized_violation ?? 0) - (a.normalized_violation ?? 0))
    .slice(0, 5);

  const satisfied = validation.satisfied && violations.length === 0;
  const maxViolationPct = Math.max(validation.max_violation || 0, ...(violations.map((v) => v.normalized_violation ?? 0)));

  return (
    <Card className="p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {satisfied ? <CheckCircle2 className="h-5 w-5 text-green-600" /> : <AlertCircle className="h-5 w-5 text-red-600" />}
          <div>
            <div className="font-medium">{satisfied ? 'Constraints satisfied' : 'Constraints violated'}</div>
            <div className="text-xs text-muted-foreground">
              Max violation: {(maxViolationPct * 100).toFixed(1)}%
            </div>
          </div>
        </div>
        <Badge variant={satisfied ? 'default' : 'destructive'} className="capitalize">
          {satisfied ? 'Pass' : 'Fail'}
        </Badge>
      </div>

      {!satisfied && violations.length > 0 && (
        <div className="space-y-2">
          {violations.map((v) => (
            <div key={v.id} className="flex items-start gap-2">
              <span className={severityBadge(v.severity)}>{v.severity}</span>
              <div className="text-sm">
            <div className="font-medium leading-tight">{v.message}</div>
            <div className="text-xs text-muted-foreground leading-tight">
              Value {formatNumber(v.metric_value)} / Threshold {formatNumber(v.threshold)}
              {v.unit ? ` ${v.unit}` : ''}
              {v.object_ids?.length ? ` • Objects: ${v.object_ids.join(', ')}` : ''}
            </div>
          </div>
        </div>
          ))}
        </div>
      )}
    </Card>
  );
}
