# Task: 3DIX Full Stack Development (Steps 9-12 + Stabilization)

## Pre-Step: Backend Stabilization
- [x] Analyze `services/api` failure and dependencies <!-- id: 0 -->
- [x] Fix `services/api` python version (switch to 3.11) and requirements <!-- id: 1 -->
- [x] Ensure `services/api` starts successfully <!-- id: 2 -->
- [x] Analyze `apps/web` server actions and auth flow <!-- id: 3 -->
- [x] Decouple `apps/web` from `services/api` runtime dependencies if any <!-- id: 4 -->
- [x] Verify `apps/web` runs without 500s <!-- id: 5 -->

## Step 9: SAM-3D Integration
- [x] Create `services/gen-sam3d-objects` microservice <!-- id: 6 -->
- [x] Implement SAM-3D logic (stubbed or actual) <!-- id: 7 -->
- [x] Integrate `services/gen-sam3d-objects` into `services/api` <!-- id: 8 -->
- [x] Implement Frontend "Replace with my furniture" flow <!-- id: 9 -->
- [x] Verify Step 9 (Scale, Round-trip, UI) <!-- id: 10 -->

## [/] **Step 10: Domain-Specific Flows**
- [x] Define config schemas in `@3dix/types` <!-- id: 11 -->
- [x] Implement constraint solver in `services/gen-sem-layout` <!-- id: 12 -->
- [x] Implement Room Design UI wizards <!-- id: 13 -->
- [x] Verify Step 10 <!-- id: 14 -->

## Step 11: SaaS Hardening
- [x] Implement Auth & Multi-tenancy <!-- id: 15 -->
- [x] Implement Billing (Stripe) <!-- id: 16 -->
- [x] Implement Analytics UI <!-- id: 17 -->
- [x] Verify Step 11 <!-- id: 18 -->

## Step 12: Deployment & Scaling
- [ ] Configure production deployment <!-- id: 19 -->
- [ ] Implement Queueing/Autoscaling <!-- id: 20 -->
- [ ] Implement Observability <!-- id: 21 -->
- [ ] Verify Step 12 <!-- id: 22 -->
