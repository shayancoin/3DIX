import { NextRequest, NextResponse } from 'next/server';
import { getTeamWithMembershipForUser, countProjectsForTeam, countRoomsForTeam } from '@/lib/db/queries';
import { hasRequiredRole } from '@/lib/auth/roles';
import { db } from '@/lib/db/drizzle';
import { layoutJobs, projects, rooms } from '@/lib/db/schema';
import { count, eq, and } from 'drizzle-orm';
import { getPlan } from '@/lib/billing/plans';
import { ensureJobCapacity } from '@/lib/billing/limits';

export async function GET(req: NextRequest) {
  try {
    const ctx = await getTeamWithMembershipForUser();
    if (!ctx) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    if (!hasRequiredRole(ctx.role, 'viewer')) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }
    const { team } = ctx;

    const [projectsCount, roomsCount, jobsTotals] = await Promise.all([
      countProjectsForTeam(team.id),
      countRoomsForTeam(team.id),
      getJobStats(team.id),
    ]);

    const plan = getPlan(team);
    const jobCapacity = await ensureJobCapacity(team);

    return NextResponse.json({
      plan: { id: plan.id, name: plan.name, limits: plan.limits },
      usage: {
        projects: projectsCount,
        rooms: roomsCount,
        jobs: jobsTotals.total,
      },
      jobs: jobsTotals,
      limits: {
        canRunJob: jobCapacity.ok,
        code: jobCapacity.ok ? null : jobCapacity.code,
        message: jobCapacity.ok ? null : jobCapacity.message,
      },
    });
  } catch (err) {
    console.error('Analytics error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

async function getJobStats(teamId: number) {
  const [totals] = await db
    .select({
      total: count(),
    })
    .from(layoutJobs)
    .leftJoin(rooms, eq(layoutJobs.roomId, rooms.id))
    .leftJoin(projects, eq(rooms.projectId, projects.id))
    .where(eq(projects.teamId, teamId));

  const [completedRow] = await db
    .select({ value: count() })
    .from(layoutJobs)
    .leftJoin(rooms, eq(layoutJobs.roomId, rooms.id))
    .leftJoin(projects, eq(rooms.projectId, projects.id))
    .where(and(eq(projects.teamId, teamId), eq(layoutJobs.status, 'completed')));

  const [failedRow] = await db
    .select({ value: count() })
    .from(layoutJobs)
    .leftJoin(rooms, eq(layoutJobs.roomId, rooms.id))
    .leftJoin(projects, eq(rooms.projectId, projects.id))
    .where(and(eq(projects.teamId, teamId), eq(layoutJobs.status, 'failed')));

  return {
    total: Number(totals?.total ?? 0),
    completed: Number(completedRow?.value ?? 0),
    failed: Number(failedRow?.value ?? 0),
  };
}
