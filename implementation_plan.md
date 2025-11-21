# 3DIX Full Stack Development Plan

## Roadmap Checklist

- [ ] Step 0: Environment Normalization & End-to-End Dev Loop <!-- id: 0 -->
    - [ ] Validate Turborepo & workspaces <!-- id: 1 -->
    - [ ] Configure global dev scripts in root `package.json` <!-- id: 2 -->
    - [ ] Configure `infra/docker-compose.dev.yml` for API <!-- id: 3 -->
    - [ ] Wire Web to API via environment variables <!-- id: 4 -->
    - [ ] Verify full dev loop (Web + API) <!-- id: 5 -->
- [ ] Step 1: Projects & Rooms Domain (UI + Persistence) <!-- id: 6 -->
    - [ ] Design DB schema for Projects and Rooms <!-- id: 7 -->
    - [ ] Implement ORM models and migrations <!-- id: 8 -->
    - [ ] Create API routes for Projects and Rooms <!-- id: 9 -->
    - [ ] Build Frontend pages for Projects and Rooms management <!-- id: 10 -->
- [ ] Step 2: Design Surface (2D Layout Canvas + Vibe Panel) <!-- id: 11 -->
    - [ ] Define client-side types (`RoomType`, `VibeSpec`, `SceneObject2D`) <!-- id: 12 -->
    - [ ] Implement 2D Layout Canvas using `react-konva` <!-- id: 13 -->
    - [ ] Build Vibe Panel (Prompt, Tags, Sliders) <!-- id: 14 -->
    - [ ] Implement Scene History stub <!-- id: 15 -->
- [ ] Step 3: Job System API (Stubbed) + Progress UI <!-- id: 16 -->
    - [ ] Create `layout_jobs` DB table <!-- id: 17 -->
    - [ ] Implement Job API endpoints (`POST`, `GET`) <!-- id: 18 -->
    - [ ] Create background stub worker <!-- id: 19 -->
    - [ ] Wire Frontend to Job API with polling and visualization <!-- id: 20 -->
- [ ] Step 4: ML Microservice Skeleton for Layout Generation <!-- id: 21 -->
    - [ ] Define ML service contract (`LayoutRequest`, `LayoutResponse`) <!-- id: 22 -->
    - [ ] Create FastAPI ML stub (`services/gen-sem-layout`) <!-- id: 23 -->
    - [ ] Integrate API service with ML service <!-- id: 24 -->
    - [ ] Update Web to use real API for layout generation <!-- id: 25 -->
- [ ] Step 5: Real SemLayoutDiff Inference <!-- id: 26 -->
    - [ ] Integrate SemLayoutDiff model in `services/gen-sem-layout` <!-- id: 27 -->
    - [ ] Map SemLayoutDiff outputs to `LayoutResponse` <!-- id: 28 -->
    - [ ] Visualize 2D semantic maps and bounding boxes in Web <!-- id: 29 -->
    - [ ] Add UI controls for conditioning (mask types) <!-- id: 30 -->
- [ ] Step 6: 3D Viewer with React Three Fiber <!-- id: 31 -->
    - [ ] Implement `LayoutScene3D` component <!-- id: 32 -->
    - [ ] Sync 2D and 3D views (selection, highlighting) <!-- id: 33 -->
- [ ] Step 7: Asset Retrieval and Mesh-based Scenes <!-- id: 34 -->
    - [ ] Prepare Asset Library (3D-FUTURE glTFs) <!-- id: 35 -->
    - [ ] Implement Asset Retrieval logic in ML service <!-- id: 36 -->
    - [ ] Implement Web Mesh Loading with quality toggle <!-- id: 37 -->
- [ ] Step 8: Vibe-coding Semantics <!-- id: 38 -->
    - [ ] Implement Vibe Encoder (Text/Image to latent) <!-- id: 39 -->
    - [ ] Implement Category Bias Learning <!-- id: 40 -->
    - [ ] Wire Vibe Panel to ML generation <!-- id: 41 -->
- [ ] Step 9: SAM-3D Integration for Custom Furniture <!-- id: 42 -->
    - [ ] Implement Object Reconstruction Service (`gen-sam3d-objects`) <!-- id: 43 -->
    - [ ] Add API endpoint for custom objects <!-- id: 44 -->
    - [ ] Build Frontend flow for object replacement <!-- id: 45 -->
- [ ] Step 10: Domain-Specific Flows <!-- id: 46 -->
    - [ ] Define Configuration Schemas for Room Types <!-- id: 47 -->
    - [ ] Implement Constraint Solver (Post-processing) <!-- id: 48 -->
    - [ ] Build UI wizards for specific room types <!-- id: 49 -->
- [ ] Step 11: SaaS Hardening <!-- id: 50 -->
    - [ ] Implement Auth and Tenant Model <!-- id: 51 -->
    - [ ] Integrate Billing (Stripe) and Quotas <!-- id: 52 -->
    - [ ] Implement Usage Logging and Analytics UI <!-- id: 53 -->
- [ ] Step 12: Deployment, Scaling & Performance <!-- id: 54 -->
    - [ ] Deploy Web App <!-- id: 55 -->
    - [ ] Deploy Backend and ML Services (GPU) <!-- id: 56 -->
    - [ ] Implement Observability and Performance Tuning <!-- id: 57 -->

---

## Step 0 • Environment Normalization & End-to-End Dev Loop

**Objective:** Achieve reproducible local development (web + API) with ≤5 min cold-start setup and ≤1 min incremental feedback loop.

**Prerequisites:** Toolchain versions (`pnpm`, `node`, `python`, `docker`), secrets scaffold, network access to ML stubs.

**Detailed Tasks**
1. **Dependency Graph Reconnaissance**
   - Run `pnpm list --depth 1` and `turbo run lint --dry-run=json` to capture turborepo task DAG.
   - Export graph artifact (`turbo graph --file turbo-graph.json`) for future CI reuse.
2. **Workspace Type Safety Audit**
   - Enforce root-level `tsconfig.base.json` inheritance; validate each package with `pnpm --filter ... lint`.
   - Prove closure by ensuring `tsconfig.references` resolves without cycle using `tsc --showConfig`.
3. **Global Script Harmonization**
   - Introduce root scripts: `dev:web`, `dev:api`, `dev:full`, `db:migrate`.
   - Document invocation complexity (worst-case O(n) packages, target <8 parallel tasks) and confirm CPU saturation <80%.
4. **API Container Baseline**
   - Normalize `docker-compose.dev.yml` to mount source (delegating to `services/api`) and apply hot-reload (`watchmedo` or `uvicorn --reload`).
   - Derive memory/CPU budget (≤2 GB RAM, ≤2 vCPUs) and validate with `docker stats`.
5. **Cross-service Contract Wiring**
   - Surface environment variables through `.env.development` → Next.js runtime and FastAPI.
   - Introduce contract tests via `pnpm --filter apps/web test:e2e -- --tags=env`.
6. **Dev Loop Proof**
   - Execute full loop: migrate DB, seed fixtures, launch compose, start web.
   - Benchmark: first paint ≤12s, API p95 latency ≤150 ms (stubbed). Record results in `/docs/metrics/dev-loop-baseline.md`.

**Acceptance Criteria**
- Deterministic bootstrap script `scripts/verify-dev-loop.sh` exits 0.
- `turbo run dev:full --concurrency=4` starts web + API without manual intervention.
- Documented metric report demonstrating performance bounds.

**Risks & Mitigations**
- *Docker networking conflicts*: reserve subnet via `docker-compose` `networks` block.
- *Monorepo drift*: gate merges with `turbo` pipeline + lockfile checksum verification.

---

## Step 1 • Projects & Rooms Domain (UI + Persistence)

**Objective:** Persist Projects/Rooms with transactional integrity; expose CRUD via API and render minimal UI with real data.

**Prerequisites:** Step 0 dev loop, database access, ORM baseline (Drizzle/Prisma equivalent).

**Detailed Tasks**
1. **Schema Derivation**
   - Normalize ER diagram: `projects (id, slug, owner_id, meta)`, `rooms (id, project_id, name, layout_state, vibe_spec)`.
   - Apply 3NF proof: ensure `layout_state` is solely dependent on `room_id`.
   - Size indexes (B-tree on `project_id`) to guarantee O(log n) lookups.
2. **Migrations & Seed**
   - Author migration set with reversible SQL; enforce deterministic ordering via timestamps.
   - Provide seed fixture (≤10 projects) for UI smoke tests.
3. **Service Layer Contracts**
   - Implement API endpoints with idempotent PUT for updates; prove atomicity with DB transaction boundaries.
   - Add OpenAPI schema and contract tests (use `schemathesis` for boundary fuzzing).
4. **Frontend Integration**
   - Generate types via `openapi-typescript` to prevent drift; compile-time guarantee by integrating into `pnpm lint`.
   - Build pages with pagination (O(k) render) and optimistic updates (bounded staleness ≤1 refresh).
5. **Observability**
   - Attach request logging (structured JSON) tagged by project id for future analytics.

**Acceptance Criteria**
- `pnpm test:api --filter services/api` passes with coverage ≥85% for domain module.
- UI list/detail views render fixture data and support create/update/delete roundtrip.

**Risks & Mitigations**
- *Schema churn*: maintain `docs/db/decision-record.md` capturing rationale, guard with migration review checklist.
- *Latency regression*: target p95 <120 ms by adding composite indexes (`project_id`, `created_at`).

---

## Step 2 • Design Surface (2D Layout Canvas + Vibe Panel)

**Objective:** Deliver performant 2D layout editor with vibe configuration storing canonical scene graph.

**Detailed Tasks**
1. **Type System Foundation**
   - Define discriminated unions for `SceneObject2D` (rectangles, polygons). Validate using Zod runtime schema; complexity O(n) validation per ingest.
2. **Canvas Rendering**
   - Implement `react-konva` stage with virtualization (only mount objects in viewport). Guarantee <16ms render budget for ≤200 shapes.
   - Add snap-to-grid (quantization proof: grid size divides coordinate space; ensures idempotent drag).
3. **Vibe Panel**
   - Compose prompt/tags/sliders into debounced state updates (≤300 ms). Provide schema validation to avoid invalid combos.
4. **Scene History Stub**
   - Implement command pattern queue with bounded size (≤50). Provide complexity O(1) push/pop, O(n) replay.

**Acceptance Criteria**
- Interaction latency measured via React Profiler under budget (<100 fps frame drop <5%).
- State persisted via React Query cache + API sync.

**Risks & Mitigations**
- *Konva memory leak*: ensure `Stage` cleanup on unmount.
- *History growth*: enforce ring buffer.

---

## Step 3 • Job System API (Stubbed) + Progress UI

**Objective:** Provide asynchronous job submission/polling for layout generation with deterministic stub worker.

**Detailed Tasks**
1. **DB Table**
   - Define `layout_jobs(id, project_id, room_id, status, payload_json, result_json, created_at, updated_at)`.
   - Indices on `(project_id, created_at)` for chronological queries.
2. **API Endpoints**
   - `POST /layout-jobs`: enqueue job (status `queued`). Validate payload ≤64KB.
   - `GET /layout-jobs/:id`: return current state; include exponential backoff hints.
3. **Stub Worker**
   - Use Celery/RQ or async task loop; simulate completion in deterministic 5s.
   - Provide invariants: exactly-once completion via status transitions (`queued→processing→succeeded/failed`).
4. **Frontend Polling**
   - Implement hook with jittered polling (start 1s, cap 10s). Provide proof of convergence to eventual stability.
5. **Progress UI**
   - Display status timeline; incorporate p95 fallback state for failure.

**Acceptance Criteria**
- Contract tests verifying state machine transitions.
- Frontend integration test verifying job progress within 6s average.

**Risks & Mitigations**
- *Race conditions*: wrap status updates in transactions.
- *Polling load*: switch to SSE/WebSocket when job volume >20 concurrent.

---

## Step 4 • ML Microservice Skeleton for Layout Generation

**Objective:** Establish FastAPI stub service with strict interface to be swapped with SemLayoutDiff inference later.

**Detailed Tasks**
1. **Contract Definition**
   - JSON schema for `LayoutRequest`, `LayoutResponse`; include versioning.
   - Add schema invariants (e.g., layout objects bounding boxes must satisfy non-negative area proof).
2. **Service Scaffolding**
   - Create FastAPI app with `/generate-layout` stub returning deterministic layout.
   - Containerize with `Dockerfile` reusing base image from Step 0.
3. **Integration**
   - API service calls ML stub; measure latency (<200 ms).
   - Add CDC (contract tests) executed via `pnpm test:contract`.
4. **Monitoring Hooks**
   - Expose Prometheus metrics (request count, latency histogram).

**Acceptance Criteria**
- End-to-end request from web yields stubbed layout.
- Metrics endpoint returns valid data.

**Risks & Mitigations**
- *Schema drift*: freeze contract version, require semver bump for breaking changes.

---

## Step 5 • Real SemLayoutDiff Inference

**Objective:** Replace stub with actual model inference; maintain deterministic fallbacks.

**Detailed Tasks**
1. **Model Integration**
   - Package SemLayoutDiff dependencies via conda lock or wheels.
   - Optimize inference graph (e.g., ONNX/TensorRT). Target <2 s per request on RTX 3090.
2. **Response Mapping**
   - Convert model output to canonical `LayoutResponse`; enforce bounding boxes in world coordinates.
3. **Visualization**
   - Render semantic maps (2D heatmaps) and bounding boxes overlays.
4. **Conditioning Controls**
   - UI toggles for mask types; ensure API payload includes `mask_mode`.
5. **Performance Validation**
   - Throughput test (≥10 req/min) with throttling.

**Acceptance Criteria**
- Benchmark report stored in `docs/metrics/semlayoutdiff.md`.
- CI smoke test runs stub path; full inference in nightly GPU pipeline.

---

## Step 6 • 3D Viewer with React Three Fiber

**Objective:** Present 3D scene synced with 2D layout; maintain 60 fps for ≤150 assets.

**Detailed Tasks**
1. **Component Scaffold**
   - Implement `LayoutScene3D` using `@react-three/fiber` + `drei`. Provide typed props referencing Step2 types.
2. **State Synchronization**
   - Shared Zustand store bridging 2D/3D selection with O(1) updates.
   - Validate referential integrity (object ids aligned).
3. **Rendering Optimizations**
   - Use instancing for repeated meshes; implement LOD toggles.
4. **Interaction**
   - Orbit controls, selection highlighting, camera fit-to-room algorithm (derived from bounding sphere calculations).

**Acceptance Criteria**
- Performance audited via React Profiler + Chrome Tracing.
- Integration test verifying selection sync.

---

## Step 7 • Asset Retrieval and Mesh-based Scenes

**Objective:** Retrieve assets from library and stream into 3D scene with adaptive quality.

**Detailed Tasks**
1. **Asset Pipeline**
   - Ingest glTF metadata into DB (precompute bounding boxes).
2. **Retrieval Logic**
   - Implement ML reranking service; prove top-k selection (k≤5) with MRR metric tracked.
3. **Web Mesh Loader**
   - Lazy load meshes with `suspense`. Provide fallback LOD toggles.
4. **Caching**
   - CDN strategy; pre-signed URLs with expiry proofs.

**Acceptance Criteria**
- Retrieval latency <500 ms.
- Mesh loads without blocking main thread (>95% frames <16ms).

---

## Step 8 • Vibe-coding Semantics

**Objective:** Map vibe inputs to latent conditioning; allow fine-tuning.

**Detailed Tasks**
1. **Vibe Encoder**
   - Implement CLIP-based embeddings; ensure cosine similarity normalization.
2. **Category Bias Learning**
   - Train small adapter; evaluate precision/recall improvements (≥10%).
3. **Panel Wiring**
   - Async inference with caching; degrade gracefully if encoder offline.

**Acceptance Criteria**
- Offline evaluation report stored in `docs/metrics/vibe-encoder.md`.
- UI updates reflect vibe selection within 500 ms.

---

## Step 9 • SAM-3D Integration for Custom Furniture

**Objective:** Allow user-provided furniture to be reconstructed and inserted into scenes.

**Detailed Tasks**
1. **Service Integration**
   - Expose `gen-sam3d-objects` with queue; ensure GPU scheduling fairness.
2. **API Endpoint**
   - Multipart upload, job tracking.
3. **Frontend Flow**
   - Replacement UI with progress + error handling.

**Acceptance Criteria**
- Upload → Reconstruction ≤5 min (p95).
- Acceptance tests verifying asset insertion.

---

## Step 10 • Domain-Specific Flows

**Objective:** Provide guided flows for room archetypes with constraint solver.

**Detailed Tasks**
1. **Config Schemas**
   - Define JSON schema for each room type; validate with AJV.
2. **Constraint Solver**
   - Implement solver (ILP or heuristic). Provide complexity analysis (target O(n log n) for n objects).
3. **UI Wizard**
   - Multi-step flows storing progress drafts.

**Acceptance Criteria**
- Constraint satisfaction tests covering edge cases.
- Wizard completion rate >90% in user tests.

---

## Step 11 • SaaS Hardening

**Objective:** Add auth, billing, analytics to operate as SaaS.

**Detailed Tasks**
1. **Auth/Tenancy**
   - Integrate with Auth0/Clerk; implement tenant isolation at DB layer (row-level security).
2. **Billing**
   - Stripe integration with webhooks; ensure idempotency.
3. **Usage Analytics**
   - Event pipeline (PostHog/Segment). Provide metrics dashboard.

**Acceptance Criteria**
- Security penetration test with zero critical findings.
- Billing integration passes test suite.

---

## Step 12 • Deployment, Scaling & Performance

**Objective:** Production-grade deployment with monitoring and tuning.

**Detailed Tasks**
1. **Deployment**
   - Infra-as-code (Terraform/Pulumi) for web + API + ML GPU nodes.
2. **Scaling**
   - Autoscaling policies (HPA for K8s). Provide load test (≥200 RPS web, ≥20 GPU jobs).
3. **Observability**
   - OpenTelemetry traces, Prometheus metrics, Grafana dashboards.

**Acceptance Criteria**
- Load test report meets SLOs.
- On-call runbook documented.

---

## Cross-Cutting Concerns

- **Testing Pyramid:** Unit (>70%), integration (>20%), E2E (critical paths).
- **CI/CD:** Turbo pipeline gating lint/test/build; nightly GPU inference regression.
- **Documentation:** Each step outputs ADR + metrics doc.
- **Security:** Static analysis (Semgrep), dependency scanning (Depfu/Snyk).

