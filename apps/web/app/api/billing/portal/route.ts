import { NextRequest, NextResponse } from 'next/server';
import { getTeamWithMembershipForUser } from '@/lib/db/queries';
import { hasRequiredRole } from '@/lib/auth/roles';
import { createCustomerPortalSession } from '@/lib/payments/stripe';

export async function POST(req: NextRequest) {
  try {
    const ctx = await getTeamWithMembershipForUser();
    if (!ctx) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    if (!hasRequiredRole(ctx.role, 'admin')) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    if (!ctx.team.stripeCustomerId) {
      return NextResponse.json({ error: 'No Stripe customer associated with team' }, { status: 400 });
    }

    const portal = await createCustomerPortalSession(ctx.team);
    return NextResponse.json({ url: portal.url });
  } catch (err) {
    console.error('Portal session error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
