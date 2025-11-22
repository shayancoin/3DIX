'use client';

import { useRouter, useParams } from 'next/navigation';
import { RoomWizard } from '@/components/design/RoomWizard';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

/**
 * Render the "Create New Room" page using the RoomWizard component.
 *
 * @returns The JSX element for the New Room page.
 */
export default function NewRoomPage() {
  const router = useRouter();
  const params = useParams();
  const projectId = params.id as string;

  return (
    <div className="container mx-auto p-6">
      <Link href={`/projects/${projectId}`} className="inline-flex items-center text-muted-foreground hover:text-foreground mb-6">
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Project
      </Link>

      <RoomWizard
        projectId={projectId}
        onCancel={() => router.back()}
      />
    </div>
  );
}
