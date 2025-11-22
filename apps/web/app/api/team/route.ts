import { getTeamWithMembershipForUser } from '@/lib/db/queries';

export async function GET() {
  const ctx = await getTeamWithMembershipForUser();
  if (!ctx) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }
  return Response.json({ team: ctx.team, role: ctx.role, user: ctx.user });
}
