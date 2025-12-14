# Multi-platform IGV Snapper container
# Supports linux/amd64 and linux/arm64
# Uses platform-independent IGV JAR + native JDK 21

FROM ubuntu:24.04

ARG IGV_VERSION=2.19.7

ENV DEBIAN_FRONTEND=noninteractive

# Install xvfb, Java 21 with full GUI support (not headless!), and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    openjdk-21-jre \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxext6 \
    libx11-6 \
    libfreetype6 \
    fontconfig \
    fonts-dejavu-core \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download platform-independent IGV (no bundled JDK)
RUN wget -q "https://data.broadinstitute.org/igv/projects/downloads/2.19/IGV_${IGV_VERSION}.zip" -O /tmp/igv.zip && \
    unzip -q /tmp/igv.zip -d / && \
    rm /tmp/igv.zip && \
    chmod +x /IGV_${IGV_VERSION}/igv.sh

# Create symlink for consistent path (matches original image structure)
RUN ln -sf /IGV_${IGV_VERSION} /IGV_Linux_2.16.2 && \
    ln -sf /IGV_${IGV_VERSION} /IGV_Linux

# Verify Java 21 is available
RUN java -version

# Create wrapper script that starts Xvfb if DISPLAY is not set
# This makes the container compatible with both:
# 1. Direct calls: igv.sh -b batch.txt (entrypoint sets up Xvfb)
# 2. xvfb-run calls: xvfb-run igv.sh -b batch.txt (xvfb-run sets up Xvfb)
COPY <<'EOF' /usr/local/bin/igv-wrapper.sh
#!/bin/bash
# If called via xvfb-run, DISPLAY will be set; if not, we need to start Xvfb
if [ -z "$DISPLAY" ]; then
    Xvfb :99 -screen 0 1024x768x24 &
    sleep 1
    export DISPLAY=:99
fi
exec "$@"
EOF
RUN chmod +x /usr/local/bin/igv-wrapper.sh

WORKDIR /work

ENTRYPOINT ["/usr/local/bin/igv-wrapper.sh"]
CMD ["/bin/bash"]
