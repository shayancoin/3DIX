import { NextRequest, NextResponse } from 'next/server';
import {
  getTeamForUser,
  getProjectWithRoomsByIdentifier,
  updateProject,
  deleteProject,
  getProjectByIdentifier,
} from '@/lib/db/queries';
import { getUser } from '@/lib/db/queries';
import { z } from 'zod';

const updateProjectSchema = z.object({
  name: z.string().min(1).max(255).optional(),
  description: z.string().optional(),
  slug: z.string().min(1).max(255).optional(),
});

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ projectSlugOrId: string }> }
) {
  try {
    const { projectSlugOrId } = await params;

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const project = await getProjectWithRoomsByIdentifier(projectSlugOrId, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    return NextResponse.json(project);
  } catch (error) {
    console.error('Error fetching project:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ projectSlugOrId: string }> }
) {
  try {
    const { projectSlugOrId } = await params;

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const existingProject = await getProjectByIdentifier(projectSlugOrId, team.id);
    if (!existingProject) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const body = await req.json();
    const validatedData = updateProjectSchema.parse(body);

    const project = await updateProject(existingProject.id, team.id, validatedData);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    return NextResponse.json(project);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      );
    }
    if (error instanceof Error && error.message.includes('Slug already in use')) {
      return NextResponse.json(
        { error: error.message },
        { status: 409 }
      );
    }
    console.error('Error updating project:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ projectSlugOrId: string }> }
) {
  try {
    const { projectSlugOrId } = await params;

    const user = await getUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const team = await getTeamForUser();
    if (!team) {
      return NextResponse.json({ error: 'No team found' }, { status: 404 });
    }

    const project = await getProjectByIdentifier(projectSlugOrId, team.id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    await deleteProject(project.id, team.id);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting project:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
