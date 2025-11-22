import { NextRequest, NextResponse } from 'next/server';
import { getProjectsForTeam, createProject, getTeamWithMembershipForUser } from '@/lib/db/queries';
import { hasRequiredRole } from '@/lib/auth/roles';
import { ensureProjectCapacity } from '@/lib/billing/limits';
import { z } from 'zod';

const createProjectSchema = z.object({
  name: z.string().min(1).max(255),
  description: z.string().optional(),
});

export async function GET() {
  try {
    const ctx = await getTeamWithMembershipForUser();
    if (!ctx) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    const team = ctx.team;
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const projects = await getProjectsForTeam(team.id);
    return NextResponse.json(projects);
  } catch (error) {
    console.error('Error fetching projects:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const ctx = await getTeamWithMembershipForUser();
    if (!ctx) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    if (!hasRequiredRole(ctx.role, 'member')) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }
    const { team, user } = ctx;

    const body = await req.json();
    const validatedData = createProjectSchema.parse(body);

    const capacity = await ensureProjectCapacity(team);
    if (!capacity.ok) {
      return NextResponse.json({ error: capacity.message, code: capacity.code }, { status: 402 });
    }

    const project = await createProject({
      teamId: team.id,
      name: validatedData.name,
      description: validatedData.description || null,
      createdBy: user.id,
    });

    return NextResponse.json(project, { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      );
    }
    console.error('Error creating project:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
