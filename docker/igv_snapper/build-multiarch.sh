#!/bin/bash
set -euo pipefail

# Build multi-platform IGV Snapper image
# Requires: docker buildx with multi-platform support
#
# This builds a native ARM64 + AMD64 IGV container using:
# - Ubuntu 24.04 base
# - OpenJDK 21 (native for each platform)
# - Platform-independent IGV JAR (no bundled JDK)
# - xvfb for headless X11

IMAGE_NAME="${1:-igv_snapper}"
IMAGE_TAG="${2:-0.1}"
PUSH="${3:-false}"

cd "$(dirname "$0")"

echo "Building multi-platform image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Create a new builder instance if needed
docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch

if [ "$PUSH" = "true" ]; then
    echo "Building and pushing to registry..."
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
        --push \
        .
else
    echo "Building for current platform..."
    docker buildx build \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
        --load \
        .
fi

echo ""
echo "Done! Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Test commands:"
echo "  # Verify Java version"
echo "  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} java -version"
echo ""
echo "  # Generate a test snapshot"
echo "  echo -e 'new\\nexit' > /tmp/test.batch"
echo "  docker run --rm -v /tmp:/tmp ${IMAGE_NAME}:${IMAGE_TAG} /IGV_Linux_2.16.2/igv.sh -b /tmp/test.batch"
echo ""
echo "Usage in ont_qc_mcp:"
echo "  export MCP_IGV_CONTAINER_IMAGE=${IMAGE_NAME}:${IMAGE_TAG}"

