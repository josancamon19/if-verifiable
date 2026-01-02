#!/bin/bash
# Publish if-verifiable to PyPI
#
# Usage:
#   ./scripts/publish.sh          # Publish to PyPI
#   ./scripts/publish.sh test     # Publish to TestPyPI first
#
# Prerequisites:
#   - uv installed
#   - PyPI API token set as TWINE_PASSWORD or in ~/.pypirc

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}📦 Building if-verifiable...${NC}"

# Clean previous builds
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Ensure build and twine are installed
uv pip install build twine --quiet

# Build the package
uv run python -m build

echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
ls -la dist/

# Check if we should publish to TestPyPI first
if [ "$1" = "test" ]; then
    echo ""
    echo -e "${YELLOW}🧪 Publishing to TestPyPI...${NC}"
    uv run python -m twine upload --repository testpypi dist/*
    echo ""
    echo -e "${GREEN}✅ Published to TestPyPI!${NC}"
    echo "Install with: pip install --index-url https://test.pypi.org/simple/ if-verifiable"
    exit 0
fi

# Confirm before publishing to real PyPI
echo ""
echo -e "${YELLOW}⚠️  About to publish to PyPI. Continue? (y/N)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo -e "${GREEN}🚀 Publishing to PyPI...${NC}"
uv run python -m twine upload dist/*

echo ""
echo -e "${GREEN}✅ Published to PyPI!${NC}"
echo "Install with: pip install if-verifiable"
