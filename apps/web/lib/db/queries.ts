import { desc, and, eq, isNull } from 'drizzle-orm';
import { db } from './drizzle';
import { activityLogs, teamMembers, teams, users, projects, rooms, ProjectWithRooms, RoomWithProject, NewProject, NewRoom, layoutJobs, LayoutJob, NewLayoutJob, JobStatus } from './schema';
import { cookies } from 'next/headers';
import { verifyToken } from '@/lib/auth/session';

export async function getUser() {
  const sessionCookie = (await cookies()).get('session');
  if (!sessionCookie || !sessionCookie.value) {
    return null;
  }

  const sessionData = await verifyToken(sessionCookie.value);
  if (
    !sessionData ||
    !sessionData.user ||
    typeof sessionData.user.id !== 'number'
  ) {
    return null;
  }

  if (new Date(sessionData.expires) < new Date()) {
    return null;
  }

  const user = await db
    .select()
    .from(users)
    .where(and(eq(users.id, sessionData.user.id), isNull(users.deletedAt)))
    .limit(1);

  if (user.length === 0) {
    return null;
  }

  return user[0];
}

export async function getTeamByStripeCustomerId(customerId: string) {
  const result = await db
    .select()
    .from(teams)
    .where(eq(teams.stripeCustomerId, customerId))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

export async function updateTeamSubscription(
  teamId: number,
  subscriptionData: {
    stripeSubscriptionId: string | null;
    stripeProductId: string | null;
    planName: string | null;
    subscriptionStatus: string;
  }
) {
  await db
    .update(teams)
    .set({
      ...subscriptionData,
      updatedAt: new Date()
    })
    .where(eq(teams.id, teamId));
}

export async function getUserWithTeam(userId: number) {
  const result = await db
    .select({
      user: users,
      teamId: teamMembers.teamId
    })
    .from(users)
    .leftJoin(teamMembers, eq(users.id, teamMembers.userId))
    .where(eq(users.id, userId))
    .limit(1);

  return result[0];
}

export async function getActivityLogs() {
  const user = await getUser();
  if (!user) {
    throw new Error('User not authenticated');
  }

  return await db
    .select({
      id: activityLogs.id,
      action: activityLogs.action,
      timestamp: activityLogs.timestamp,
      ipAddress: activityLogs.ipAddress,
      userName: users.name
    })
    .from(activityLogs)
    .leftJoin(users, eq(activityLogs.userId, users.id))
    .where(eq(activityLogs.userId, user.id))
    .orderBy(desc(activityLogs.timestamp))
    .limit(10);
}

export async function getTeamForUser() {
  const user = await getUser();
  if (!user) {
    return null;
  }

  const result = await db.query.teamMembers.findFirst({
    where: eq(teamMembers.userId, user.id),
    with: {
      team: {
        with: {
          teamMembers: {
            with: {
              user: {
                columns: {
                  id: true,
                  name: true,
                  email: true
                }
              }
            }
          }
        }
      }
    }
  });

  return result?.team || null;
}

// Projects & Rooms Queries

export async function getProjectsForTeam(teamId: number) {
  return await db
    .select()
    .from(projects)
    .where(and(
      eq(projects.teamId, teamId),
      isNull(projects.deletedAt)
    ))
    .orderBy(desc(projects.updatedAt));
}

export async function getProjectWithRooms(projectId: number, teamId: number): Promise<ProjectWithRooms | null> {
  const project = await db
    .select()
    .from(projects)
    .where(and(
      eq(projects.id, projectId),
      eq(projects.teamId, teamId),
      isNull(projects.deletedAt)
    ))
    .limit(1);

  if (project.length === 0) {
    return null;
  }

  const projectRooms = await db
    .select()
    .from(rooms)
    .where(and(
      eq(rooms.projectId, projectId),
      isNull(rooms.deletedAt)
    ))
    .orderBy(desc(rooms.updatedAt));

  return {
    ...project[0],
    rooms: projectRooms,
  };
}

export async function createProject(data: NewProject) {
  const [project] = await db
    .insert(projects)
    .values(data)
    .returning();

  return project;
}

export async function updateProject(projectId: number, teamId: number, data: Partial<NewProject>) {
  const [project] = await db
    .update(projects)
    .set({
      ...data,
      updatedAt: new Date(),
    })
    .where(and(
      eq(projects.id, projectId),
      eq(projects.teamId, teamId),
      isNull(projects.deletedAt)
    ))
    .returning();

  return project;
}

export async function deleteProject(projectId: number, teamId: number) {
  await db
    .update(projects)
    .set({
      deletedAt: new Date(),
      updatedAt: new Date(),
    })
    .where(and(
      eq(projects.id, projectId),
      eq(projects.teamId, teamId),
      isNull(projects.deletedAt)
    ));
}

export async function getRoom(roomId: number, projectId: number): Promise<RoomWithProject | null> {
  const room = await db
    .select()
    .from(rooms)
    .where(and(
      eq(rooms.id, roomId),
      eq(rooms.projectId, projectId),
      isNull(rooms.deletedAt)
    ))
    .limit(1);

  if (room.length === 0) {
    return null;
  }

  const project = await db
    .select()
    .from(projects)
    .where(eq(projects.id, projectId))
    .limit(1);

  if (project.length === 0) {
    return null;
  }

  return {
    ...room[0],
    project: project[0],
  };
}

export async function getRoomsForProject(projectId: number) {
  return await db
    .select()
    .from(rooms)
    .where(and(
      eq(rooms.projectId, projectId),
      isNull(rooms.deletedAt)
    ))
    .orderBy(desc(rooms.updatedAt));
}

export async function createRoom(data: NewRoom) {
  const [room] = await db
    .insert(rooms)
    .values(data)
    .returning();

  return room;
}

export async function updateRoom(roomId: number, projectId: number, data: Partial<NewRoom>) {
  const [room] = await db
    .update(rooms)
    .set({
      ...data,
      updatedAt: new Date(),
    })
    .where(and(
      eq(rooms.id, roomId),
      eq(rooms.projectId, projectId),
      isNull(rooms.deletedAt)
    ))
    .returning();

  return room;
}

export async function deleteRoom(roomId: number, projectId: number) {
  await db
    .update(rooms)
    .set({
      deletedAt: new Date(),
      updatedAt: new Date(),
    })
    .where(and(
      eq(rooms.id, roomId),
      eq(rooms.projectId, projectId),
      isNull(rooms.deletedAt)
    ));
}

// Layout Jobs Queries

export async function createLayoutJob(data: NewLayoutJob) {
  const [job] = await db
    .insert(layoutJobs)
    .values(data)
    .returning();

  return job;
}

export async function getLayoutJob(jobId: number) {
  const [job] = await db
    .select()
    .from(layoutJobs)
    .where(eq(layoutJobs.id, jobId))
    .limit(1);

  return job || null;
}

export async function getLayoutJobsForRoom(roomId: number) {
  return await db
    .select()
    .from(layoutJobs)
    .where(eq(layoutJobs.roomId, roomId))
    .orderBy(desc(layoutJobs.createdAt));
}

export async function getLatestLayoutJobForRoom(roomId: number) {
  const [job] = await db
    .select()
    .from(layoutJobs)
    .where(eq(layoutJobs.roomId, roomId))
    .orderBy(desc(layoutJobs.createdAt))
    .limit(1);

  return job || null;
}

export async function updateLayoutJob(jobId: number, updates: Partial<NewLayoutJob>) {
  const [job] = await db
    .update(layoutJobs)
    .set({
      ...updates,
      updatedAt: new Date(),
    })
    .where(eq(layoutJobs.id, jobId))
    .returning();

  return job;
}

export async function getQueuedJobs(limit: number = 10) {
  return await db
    .select()
    .from(layoutJobs)
    .where(eq(layoutJobs.status, JobStatus.QUEUED))
    .orderBy(layoutJobs.createdAt)
    .limit(limit);
}

export async function getRunningJobs() {
  return await db
    .select()
    .from(layoutJobs)
    .where(eq(layoutJobs.status, JobStatus.RUNNING));
}
