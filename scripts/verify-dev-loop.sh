#!/bin/bash
# Verification script for the development loop
# This script checks that both web and API services can be started

set -e

echo "🔍 Verifying Development Loop Setup..."
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pnpm is installed
echo "Checking pnpm..."
if ! command -v pnpm &> /dev/null; then
    echo -e "${RED}✗ pnpm is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ pnpm is installed${NC}"

# Check if Python is installed
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 is installed${NC}"

# Check if Docker is installed (optional)
echo "Checking Docker (optional)..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker is installed${NC}"
else
    echo -e "${YELLOW}⚠ Docker is not installed (optional for API development)${NC}"
fi

# Check workspace configuration
echo "Checking workspace configuration..."
if [ -f "pnpm-workspace.yaml" ]; then
    echo -e "${GREEN}✓ pnpm-workspace.yaml exists${NC}"
else
    echo -e "${RED}✗ pnpm-workspace.yaml not found${NC}"
    exit 1
fi

# Check turbo.json
if [ -f "turbo.json" ]; then
    echo -e "${GREEN}✓ turbo.json exists${NC}"
else
    echo -e "${RED}✗ turbo.json not found${NC}"
    exit 1
fi

# Check environment files
echo "Checking environment files..."
if [ -f "apps/web/.env.example" ]; then
    echo -e "${GREEN}✓ apps/web/.env.example exists${NC}"
else
    echo -e "${YELLOW}⚠ apps/web/.env.example not found${NC}"
fi

if [ -f "apps/web/.env.development" ]; then
    echo -e "${GREEN}✓ apps/web/.env.development exists${NC}"
else
    echo -e "${YELLOW}⚠ apps/web/.env.development not found${NC}"
fi

if [ -f "services/api/.env.dev" ]; then
    echo -e "${GREEN}✓ services/api/.env.dev exists${NC}"
else
    echo -e "${YELLOW}⚠ services/api/.env.dev not found${NC}"
fi

# Check docker-compose
if [ -f "infra/docker-compose.dev.yml" ]; then
    echo -e "${GREEN}✓ infra/docker-compose.dev.yml exists${NC}"
else
    echo -e "${YELLOW}⚠ infra/docker-compose.dev.yml not found${NC}"
fi

# Check API client
if [ -f "apps/web/lib/api/client.ts" ]; then
    echo -e "${GREEN}✓ API client exists${NC}"
else
    echo -e "${YELLOW}⚠ API client not found${NC}"
fi

echo ""
echo -e "${GREEN}✅ Development loop setup verification complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Configure environment variables (see README.md)"
echo "2. Install dependencies: pnpm install"
echo "3. Start backend: pnpm dev:api (or pnpm dev:api:docker)"
echo "4. Start frontend: pnpm dev:web"
echo "5. Use pnpm dev:stack for a combined loop when both services are needed"
echo "6. Visit http://localhost:3000 (web) and http://localhost:8000/docs (API)"
