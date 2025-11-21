import { NextRequest, NextResponse } from 'next/server';
import { getTeamForUser, getProjectsForTeam, createProject, generateProjectSlug } from '@/lib/db/queries';
import { getUser } from '@/lib/db/queries';
import { z } from 'zod';

const createProjectSchema = z.object({
  name: z.string().min(1).max(255),
  description: z.string().optional(),
  slug: z.string().min(1).max(255).optional(),
});

export async function GET() {
  try {
    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
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
    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const body = await req.json();
    const validatedData = createProjectSchema.parse(body);

    const slug = await generateProjectSlug(team.id, validatedData.slug || validatedData.name);

    const project = await createProject({
      teamId: team.id,
      name: validatedData.name,
      description: validatedData.description || null,
      slug,
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
