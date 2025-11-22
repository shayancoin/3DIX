import { Team } from '@/lib/db/schema';
import { getPlan } from './plans';
import { countProjectsForTeam, countRoomsForTeam, countTeamJobsInMonth } from '@/lib/db/queries';

export type LimitCode = 'PROJECT_LIMIT' | 'ROOM_LIMIT' | 'JOB_LIMIT';

export type LimitCheckResult =
  | { ok: true }
  | { ok: false; code: LimitCode; message: string };

async function checkProjects(team: Team, plan = getPlan(team)): Promise<LimitCheckResult> {
  const projectCount = await countProjectsForTeam(team.id);
  if (projectCount >= plan.limits.projects) {
    return { ok: false, code: 'PROJECT_LIMIT', message: `Project limit reached (${plan.limits.projects}). Upgrade to create more projects.` };
  }
  return { ok: true };
}

async function checkRooms(team: Team, plan = getPlan(team)): Promise<LimitCheckResult> {
  const roomCount = await countRoomsForTeam(team.id);
  if (roomCount >= plan.limits.rooms) {
    return { ok: false, code: 'ROOM_LIMIT', message: `Room limit reached (${plan.limits.rooms}). Upgrade to create more rooms.` };
  }
  return { ok: true };
}

async function checkJobs(team: Team, plan = getPlan(team)): Promise<LimitCheckResult> {
  const now = new Date();
  const startOfMonth = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  const endOfMonth = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 0, 23, 59, 59, 999));
  const jobsThisMonth = await countTeamJobsInMonth(team.id, startOfMonth, endOfMonth);

  if (jobsThisMonth >= plan.limits.jobsPerMonth) {
    return { ok: false, code: 'JOB_LIMIT', message: `Job limit reached for this month (${plan.limits.jobsPerMonth}).` };
  }
  return { ok: true };
}

export async function ensureProjectCapacity(team: Team): Promise<LimitCheckResult> {
  const plan = getPlan(team);
  return checkProjects(team, plan);
}

export async function ensureRoomCapacity(team: Team): Promise<LimitCheckResult> {
  const plan = getPlan(team);
  return checkRooms(team, plan);
}

export async function ensureJobCapacity(team: Team): Promise<LimitCheckResult> {
  const plan = getPlan(team);
  return checkJobs(team, plan);
}
