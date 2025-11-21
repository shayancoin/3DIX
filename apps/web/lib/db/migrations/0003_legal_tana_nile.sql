ALTER TABLE "projects" ADD COLUMN "slug" varchar(255);--> statement-breakpoint
UPDATE "projects" SET "slug" = CONCAT('project-', "id") WHERE "slug" IS NULL;--> statement-breakpoint
ALTER TABLE "projects" ALTER COLUMN "slug" SET NOT NULL;--> statement-breakpoint
ALTER TABLE "rooms" ADD COLUMN "floorplan_url" text;--> statement-breakpoint
CREATE UNIQUE INDEX "projects_team_slug_unique" ON "projects" USING btree ("team_id","slug");