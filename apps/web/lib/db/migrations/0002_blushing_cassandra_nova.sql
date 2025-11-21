CREATE TABLE "layout_jobs" (
	"id" serial PRIMARY KEY NOT NULL,
	"room_id" integer NOT NULL,
	"status" varchar(20) DEFAULT 'queued' NOT NULL,
	"request_data" jsonb,
	"response_data" jsonb,
	"progress" integer DEFAULT 0,
	"progress_message" text,
	"error_message" text,
	"error_details" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"started_at" timestamp,
	"completed_at" timestamp,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	"worker_id" varchar(255),
	"retry_count" integer DEFAULT 0
);
--> statement-breakpoint
ALTER TABLE "layout_jobs" ADD CONSTRAINT "layout_jobs_room_id_rooms_id_fk" FOREIGN KEY ("room_id") REFERENCES "public"."rooms"("id") ON DELETE cascade ON UPDATE no action;