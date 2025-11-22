import { NextRequest, NextResponse } from 'next/server';
import { getTeamWithMembershipForUser } from '@/lib/db/queries';
import { hasRequiredRole } from '@/lib/auth/roles';
import { stripe } from '@/lib/payments/stripe';
import { getPlan, getPriceIdForPlan, PlanId, PLANS } from '@/lib/billing/plans';

const bodySchema = {
  parse(input: any) {
    const planId = typeof input?.planId === 'string' ? (input.planId as PlanId) : undefined;
    const priceId = typeof input?.priceId === 'string' ? input.priceId : undefined;
    return { planId, priceId };
  },
};

export async function POST(req: NextRequest) {
  try {
    const ctx = await getTeamWithMembershipForUser();
    if (!ctx) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    if (!hasRequiredRole(ctx.role, 'admin')) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }
    const { team, user } = ctx;

    let raw: any = {};
    try {
      raw = await req.json();
    } catch {
      raw = {};
    }
    const body = bodySchema.parse(raw);
    const plan = body.planId ? PLANS[body.planId] : getPlan(team);
    const priceId = body.priceId || getPriceIdForPlan(plan.id);

    if (!priceId) {
      return NextResponse.json({ error: 'Missing Stripe price for plan' }, { status: 400 });
    }

    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      client_reference_id: user.id.toString(),
      mode: 'subscription',
      customer: team.stripeCustomerId || undefined,
      success_url: `${process.env.BASE_URL}/billing/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.BASE_URL}/billing/cancel`,
      line_items: [{ price: priceId, quantity: 1 }],
      metadata: { teamId: team.id.toString() },
      subscription_data: {
        trial_period_days: plan.id === 'free' ? undefined : 14,
      },
    });

    return NextResponse.json({ url: session.url });
  } catch (err) {
    console.error('Checkout session error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
