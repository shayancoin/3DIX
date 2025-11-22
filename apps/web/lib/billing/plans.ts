import { Team } from '@/lib/db/schema';

export type PlanId = 'free' | 'pro' | 'pro_plus';

export interface PlanLimits {
  projects: number;
  rooms: number;
  jobsPerMonth: number;
  members: number;
}

export interface PlanConfig {
  id: PlanId;
  name: string;
  limits: PlanLimits;
  priceIdEnv: string;
}

export const PLANS: Record<PlanId, PlanConfig> = {
  free: {
    id: 'free',
    name: 'Free',
    limits: {
      projects: 2,
      rooms: 10,
      jobsPerMonth: 25,
      members: 3,
    },
    priceIdEnv: 'STRIPE_PRICE_ID_FREE',
  },
  pro: {
    id: 'pro',
    name: 'Pro',
    limits: {
      projects: 10,
      rooms: 100,
      jobsPerMonth: 300,
      members: 10,
    },
    priceIdEnv: 'STRIPE_PRICE_ID_PRO',
  },
  pro_plus: {
    id: 'pro_plus',
    name: 'Pro+',
    limits: {
      projects: 50,
      rooms: 500,
      jobsPerMonth: 2000,
      members: 25,
    },
    priceIdEnv: 'STRIPE_PRICE_ID_PRO_PLUS',
  },
};

export function resolvePlanId(team: Team | null): PlanId {
  const planName = team?.planName?.toLowerCase() ?? 'free';
  if (planName.includes('pro+')) return 'pro_plus';
  if (planName.includes('pro')) return 'pro';
  return 'free';
}

export function getPlan(team: Team | null): PlanConfig {
  const id = resolvePlanId(team);
  return PLANS[id];
}

export function getPriceIdForPlan(plan: PlanId): string | undefined {
  const envKey = PLANS[plan].priceIdEnv;
  return process.env[envKey];
}
