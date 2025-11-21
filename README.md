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

## Step 0: Environment Setup

### Prerequisites

- **Node.js** 18+ and **pnpm** 9.1.0+
- **Python** 3.10+
- **Docker** (optional, for containerized development)
- **Stripe CLI** (for web app payment integration)

### Initial Setup

1. **Install dependencies:**
   ```bash
   # Install frontend dependencies
   pnpm install
   
   # Install backend dependencies
   cd services/api
   pip install -r requirements.txt
   cd ../..
   ```

2. **Configure environment variables:**
   
   **Web App (apps/web):**
   ```bash
   cd apps/web
   # Use the setup script to create .env file
   pnpm db:setup
    # Or manually copy and configure
   cp .env.example .env.local
   ```
    
    > ℹ️ The repo includes `apps/web/.env.development` with localhost-safe defaults
    > (`API_URL`/`NEXT_PUBLIC_API_URL` pointing to `http://localhost:8000`). Next.js
    > loads it automatically during `pnpm dev:web`, so you only need to create
    > `.env.local` when overriding sensitive values (Stripe keys, database, etc.).
   
   Required variables:
   - `POSTGRES_URL`: Database connection string
   - `STRIPE_SECRET_KEY`: Stripe API secret key
   - `STRIPE_WEBHOOK_SECRET`: Stripe webhook secret (run `stripe listen --print-secret`)
   - `BASE_URL`: Application base URL (default: `http://localhost:3000`)
   - `AUTH_SECRET`: Random secret for JWT (generate with `openssl rand -base64 32`)
   
   **API Service (services/api):**
   ```bash
   cd services/api
   # Copy the example file
   cp .env.example .env.dev
   # Edit .env.dev with your configuration
   ```
   
   Required variables:
   - `ENV_STATE`: Environment state (`dev` or `prod`)
   - `DEV_API_NAME`, `DEV_API_DESCRIPTION`, `DEV_API_VERSION`: API metadata
   - `DEV_HOST`, `DEV_PORT`: Server configuration
   - `DEV_LOG_LEVEL`: Logging level

3. **Database setup (Web App):**
   ```bash
   cd apps/web
   # Run migrations
   pnpm db:migrate
   # Seed database (optional)
   pnpm db:seed
   ```

### Getting Started

**Frontend (Development):**
```bash
# From root directory
pnpm dev:web
# Or from apps/web
cd apps/web && pnpm dev
```

**Backend (Development):**

Option 1: Direct Python execution
```bash
# From root directory
pnpm dev:api
# Or from services/api
cd services/api
uvicorn manage:app --reload --host 0.0.0.0 --port 8000
```

Option 2: Docker Compose (with hot reload)
```bash
# From root directory
pnpm dev:api:docker
```

The API will be available at `http://localhost:8000` and the web app at `http://localhost:3000`.

**Full Stack Loop (Web + API):**
```bash
pnpm dev:stack
```

**ML Service (Development):**
```bash
# From services/gen-sem-layout
cd services/gen-sem-layout
pip install -r requirements.txt
python main.py
# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The ML service will be available at `http://localhost:8001`.

### Development Workflow

- **Hot Reload**: Both frontend and backend support hot reload
  - Frontend: Next.js automatically reloads on file changes
  - Backend: Uvicorn with `--reload` flag watches for Python file changes
- **Docker Development**: Use `docker-compose.dev.yml` for containerized development with volume mounts for live code updates

### Projects & Rooms Domain

- Projects receive stable, per-team slugs (auto-generated from the name) so sharing a link keeps working even if IDs change.
- Dashboard routes and API endpoints accept either the slug or numeric ID, but all newly generated links use the slug.
- Room records capture optional `floorplanUrl` references, enabling floor-plan–conditioned jobs in later steps.
- The dashboard room forms expose a floor plan URL field; room detail pages surface the reference for quick verification.

### Verifying the Dev Loop

1. **Start the Backend API:**
   ```bash
   # Option 1: Direct Python execution
   pnpm dev:api
   
   # Option 2: Docker Compose
   pnpm dev:api:docker
   ```
   Verify: Visit `http://localhost:8000/docs` to see the FastAPI Swagger UI

2. **Start the Frontend:**
   ```bash
   pnpm dev:web
   
   # Or start both services from one terminal
   pnpm dev:stack
   ```
   Verify: Visit `http://localhost:3000` to see the web app

3. **Test API Integration:**
   - The web app is configured to call the backend API at `http://localhost:8000`
   - The `/api/vibe/echo` endpoint in the web app proxies to the backend API
   - Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in your `.env.local` for client-side API calls

4. **Automated verification:**
   ```bash
   pnpm verify:devloop
   ```

### Environment Variables Summary

**Web App (`apps/web/.env.local`):**
- `POSTGRES_URL`: Database connection
- `STRIPE_SECRET_KEY`: Stripe API key
- `STRIPE_WEBHOOK_SECRET`: Stripe webhook secret
- `BASE_URL`: Web app base URL (default: `http://localhost:3000`)
- `AUTH_SECRET`: JWT secret
- `API_URL`: Server-side backend API URL (default: `http://localhost:8000`)
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: `http://localhost:8000`)
  - Pre-populated via `apps/web/.env.development` for local development

**API Service (`services/api/.env.dev`):**
- `ENV_STATE`: Environment state (`dev` or `prod`)
- `DEV_API_NAME`, `DEV_API_DESCRIPTION`, `DEV_API_VERSION`: API metadata
- `DEV_HOST`, `DEV_PORT`: Server configuration (default: `0.0.0.0:8000`)
- `DEV_LOG_LEVEL`: Logging level (default: `debug`)

