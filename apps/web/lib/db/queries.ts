import { desc, and, eq, isNull } from 'drizzle-orm';
import { db } from './drizzle';
import { activityLogs, teamMembers, teams, users, projects, rooms, ProjectWithRooms, RoomWithProject, NewProject, NewRoom, layoutJobs, LayoutJob, NewLayoutJob, JobStatus as JobStatusEnum } from './schema';
import { cookies } from 'next/headers';
import { verifyToken } from '@/lib/auth/session';
import { assertValidTransition, JobStatus } from '@/lib/jobs/stateMachine';

/**
 * Retrieve the currently authenticated user based on the session cookie.
 *
 * Returns the user record associated with a valid, non-expired session; returns `null` if the session is missing, invalid, expired, or the user does not exist or is deleted.
 *
 * @returns The authenticated user's record, or `null` when no authenticated user is available.
 */
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

/**
 * Retrieve a room by its ID together with its parent project.
 *
 * @param roomId - The ID of the room to fetch
 * @param projectId - The ID of the project the room must belong to
 * @returns The room object with an added `project` property when both the room and project exist, `null` otherwise
 */
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

/**
 * Fetches a room by ID and includes its associated project when the project is not deleted.
 *
 * @returns The room object with its `project` when both exist and the project is not deleted, or `null` otherwise.
 */
export async function getRoomWithProjectByRoomId(roomId: number): Promise<RoomWithProject | null> {
  const room = await db.query.rooms.findFirst({
    where: and(eq(rooms.id, roomId), isNull(rooms.deletedAt)),
    with: {
      project: true,
    },
  });

  if (!room || !room.project || room.project.deletedAt) {
    return null;
  }

  return {
    ...room,
    project: room.project,
  };
}

/**
 * Retrieve non-deleted rooms for a project, ordered by most recently updated first.
 *
 * @param projectId - The project's numeric identifier
 * @returns An array of room records belonging to the specified project, ordered by `updatedAt` descending
 */
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

export type CreateLayoutJobInput = {
  roomId: number;
  requestData: unknown;
  status?: JobStatus;
  progress?: number | null;
  progressMessage?: string | null;
};

export type UpdateLayoutJobInput = Partial<Pick<NewLayoutJob,
  'requestData' |
  'responseData' |
  'errorMessage' |
  'errorDetails' |
  'workerId' |
  'retryCount'
>> & {
  status?: JobStatus;
  progress?: number | null;
  progressMessage?: string | null;
  startedAt?: Date | null;
  completedAt?: Date | null;
};

/**
 * Create a new layout job record for a room.
 *
 * @param data - Input properties for the new layout job: `roomId`, `requestData`, and optional `status`, `progress`, and `progressMessage`
 * @returns The newly created layout job record
 */
export async function createLayoutJob(data: CreateLayoutJobInput) {
  const now = new Date();
  const [job] = await db
    .insert(layoutJobs)
    .values({
      roomId: data.roomId,
      requestData: data.requestData,
      status: (data.status ?? JobStatusEnum.QUEUED) as JobStatusEnum,
      progress: data.progress ?? 0,
      progressMessage: data.progressMessage ?? 'Queued',
      createdAt: now,
      updatedAt: now,
    })
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

/**
 * Fetches the most recently created layout job for a given room.
 *
 * @returns The most recently created layout job for the room, or `null` if none exists.
 */
export async function getLatestLayoutJobForRoom(roomId: number) {
  const [job] = await db
    .select()
    .from(layoutJobs)
    .where(eq(layoutJobs.roomId, roomId))
    .orderBy(desc(layoutJobs.createdAt))
    .limit(1);

  return job || null;
}

/**
 * Update a layout job with the given fields and ensure any status change is valid.
 *
 * @param jobId - The ID of the layout job to update
 * @param updates - Partial fields to apply to the job; when `status` is provided, the transition is validated
 * @returns The updated layout job record, or `null` if no job with `jobId` exists
 * @throws If `updates.status` is provided and the status transition from the existing job is invalid
 */
export async function updateLayoutJob(jobId: number, updates: UpdateLayoutJobInput) {
  const existing = await getLayoutJob(jobId);
  if (!existing) {
    return null;
  }

  if (updates.status) {
    assertValidTransition(existing.status as JobStatus, updates.status);
  }

  const [job] = await db
    .update(layoutJobs)
    .set({
      ...updates,
      status: updates.status ?? existing.status,
      updatedAt: new Date(),
    })
    .where(eq(layoutJobs.id, jobId))
    .returning();

  return job || null;
}

/**
 * Fetches queued layout jobs ordered by creation time (oldest first).
 *
 * @param limit - Maximum number of jobs to return (defaults to 10)
 * @returns An array of layout job records with status `QUEUED`, ordered by `createdAt`, limited to `limit`
 */
export async function getQueuedJobs(limit: number = 10) {
  return await db
    .select()
    .from(layoutJobs)
    .where(eq(layoutJobs.status, JobStatusEnum.QUEUED))
    .orderBy(layoutJobs.createdAt)
    .limit(limit);
}

/**
 * Fetches layout jobs that are currently running.
 *
 * @returns An array of layout job records whose status is `RUNNING`.
 */
export async function getRunningJobs() {
  return await db
    .select()
    .from(layoutJobs)
    .where(eq(layoutJobs.status, JobStatusEnum.RUNNING));
}