# Job System Verification

Manual E2E (until Playwright is wired):

1) Start services: `pnpm dev` in `apps/web`, `pnpm layout-worker` (requires `POSTGRES_URL`), and the ML stub `cd services/gen-sem-layout && uvicorn main:app --host 0.0.0.0 --port 8001` (or via docker-compose).
2) Seed a user/team/project/room (via UI or SQL). Note the room id.
3) Visit `/projects/{projectId}/rooms/{roomId}`; click **Generate layout**.
4) Within a few seconds the status card should move `queued → running → completed`, and the canvas should render at least one object generated from the stub layout.
5) Inspect `layout_jobs` in Postgres to confirm `response_data` is persisted and status is terminal.

Automated tests:
- `apps/web/tests/jobStateMachine.test.ts` exercises the status FSM and stub result shape via Node’s built-in test runner (run with `cd apps/web && node --test --loader ts-node/register ./tests/jobStateMachine.test.ts`).
