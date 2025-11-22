import { Team } from '@/lib/db/schema';
import { getTeamWithMembershipForUser } from '@/lib/db/queries';

export type Role = 'viewer' | 'member' | 'admin' | 'owner';

const ROLE_ORDER: Record<Role, number> = {
  viewer: 0,
  member: 1,
  admin: 2,
  owner: 3,
};

export function hasRequiredRole(current: string | null | undefined, minRole: Role): boolean {
  if (!current) return false;
  const normalized = (current.toLowerCase() as Role) || 'viewer';
  return (ROLE_ORDER[normalized] ?? 0) >= ROLE_ORDER[minRole];
}

export async function getTeamContext(minRole: Role = 'member') {
  const context = await getTeamWithMembershipForUser();
  if (!context) {
    return null;
  }
  if (!hasRequiredRole(context.role, minRole)) {
    return null;
  }
  return context as { user: { id: number }; team: Team; role: string };
}
