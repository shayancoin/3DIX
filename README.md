# 3DIX

3DIX is a full-stack SaaS application for "vibe-coding" automatic 3D indoor environment generation. It leverages a Next.js frontend, a Python FastAPI backend, and research-grade generative AI models (SemLayoutDiff, SAM-3D) to create interactive 3D kitchen, closet, and bathroom designs from natural language prompts.

## Repo Layout

- **apps/**: Application entry points.
  - `web`: Main Next.js SaaS frontend (React Three Fiber + Chatbot).
- **services/**: Backend services.
  - `api`: Python FastAPI orchestration gateway.
  - `gen-*`: Stub services for generative AI models.
- **packages/**: Shared libraries.
  - `ui`: Shared React UI components.
  - `three`: Shared React Three Fiber components.
  - `config`: Shared configurations (TS, ESLint, etc.).
  - `types`: Shared TypeScript types.
- **research/**: Research codebases (SemLayoutDiff, SAM-3D) integrated as read-only references.
- **infra/**: Infrastructure configuration (Docker, scripts).

## Pre-Step #0 – Canonical Fusion

This monorepo was fused from the following sources:
- **apps/web**: `saas-starter` + `r3f-next-starter` + `ai-chatbot` (UI).
- **services/api**: `fastapi-template`.
- **research/**: `SemLayoutDiff`, `sam-3d-objects`, `sam-3d-body`.

### Getting Started

**Frontend:**
```bash
pnpm install
pnpm dev --filter @3dix/web
```

**Backend:**
```bash
cd services/api
# Ensure python 3.10+ and dependencies
pip install -r requirements.txt
uvicorn manage:app --reload
```

