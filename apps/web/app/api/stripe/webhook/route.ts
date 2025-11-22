import Stripe from 'stripe';
import { handleSubscriptionChange, stripe } from '@/lib/payments/stripe';
import { NextRequest, NextResponse } from 'next/server';
import { updateTeamSubscription, getTeamByStripeCustomerId } from '@/lib/db/queries';

const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!;

export async function POST(request: NextRequest) {
  const payload = await request.text();
  const signature = request.headers.get('stripe-signature') as string;

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(payload, signature, webhookSecret);
  } catch (err) {
    console.error('Webhook signature verification failed.', err);
    return NextResponse.json(
      { error: 'Webhook signature verification failed.' },
      { status: 400 }
    );
  }

  switch (event.type) {
    case 'checkout.session.completed': {
      const sessionObj = event.data.object as Stripe.Checkout.Session;
      const subscriptionId = typeof sessionObj.subscription === 'string' ? sessionObj.subscription : sessionObj.subscription?.id;
      const customerId = typeof sessionObj.customer === 'string' ? sessionObj.customer : sessionObj.customer?.id;
      if (subscriptionId && customerId) {
        const subscription = await stripe.subscriptions.retrieve(subscriptionId, { expand: ['items.data.price.product'] });
        const team = await getTeamByStripeCustomerId(customerId);
        if (team) {
          const plan = subscription.items.data[0]?.plan;
          await updateTeamSubscription(team.id, {
            stripeSubscriptionId: subscription.id,
            stripeProductId: plan?.product as string,
            planName: (plan?.product as Stripe.Product).name,
            subscriptionStatus: subscription.status
          });
        }
      }
      break;
    }
    case 'customer.subscription.updated':
    case 'customer.subscription.deleted':
      const subscription = event.data.object as Stripe.Subscription;
      await handleSubscriptionChange(subscription);
      break;
    default:
      console.log(`Unhandled event type ${event.type}`);
  }

  return NextResponse.json({ received: true });
}
