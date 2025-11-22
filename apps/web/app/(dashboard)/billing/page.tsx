'use client';

import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

type TeamResponse = { team?: { planName?: string | null }; role?: string; error?: string };

export default function BillingPage() {
  const [teamInfo, setTeamInfo] = useState<TeamResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/team')
      .then((res) => res.json())
      .then(setTeamInfo)
      .catch((err) => setError(err.message));
  }, []);

  const startCheckout = async () => {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const res = await fetch('/api/billing/checkout', { method: 'POST' });
      const json = await res.json();
      if (!res.ok) {
        setError(json.error || 'Failed to start checkout');
      } else if (json.url) {
        window.location.href = json.url;
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const openPortal = async () => {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const res = await fetch('/api/billing/portal', { method: 'POST' });
      const json = await res.json();
      if (!res.ok) {
        setError(json.error || 'Failed to open portal');
      } else if (json.url) {
        window.location.href = json.url;
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Billing</h1>
          <p className="text-muted-foreground text-sm">Manage your subscription</p>
        </div>
        <Badge>{teamInfo?.team?.planName || 'Free'}</Badge>
      </div>

      <Card className="p-4 space-y-3">
        <div className="flex items-center gap-2">
          <div className="font-semibold">Subscription</div>
          {teamInfo?.role && <Badge variant="outline" className="capitalize">Role: {teamInfo.role}</Badge>}
        </div>
        <div className="flex gap-2">
          <Button onClick={startCheckout} disabled={loading}>
            {loading ? 'Starting...' : 'Upgrade / Change plan'}
          </Button>
          <Button variant="outline" onClick={openPortal} disabled={loading}>
            Manage billing
          </Button>
        </div>
        {message && <div className="text-sm text-green-600">{message}</div>}
        {error && <div className="text-sm text-destructive">{error}</div>}
      </Card>
    </div>
  );
}
