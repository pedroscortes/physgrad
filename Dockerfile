# Multi-stage build for PhysGrad cloud deployment
FROM ubuntu:22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy source code
COPY . .

# Build PhysGrad
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON && \
    make -j$(nproc)

# Production stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 physgrad && \
    mkdir -p /app/data /app/logs && \
    chown -R physgrad:physgrad /app

# Copy built binaries
COPY --from=builder /app/build/bin/* /app/bin/
COPY --from=builder /app/build/lib/* /app/lib/
COPY --from=builder /app/src/ /app/src/
COPY --from=builder /app/test_* /app/

# Copy deployment scripts
COPY deployment/ /app/deployment/

# Set library path
ENV LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH

# Switch to non-root user
USER physgrad
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/deployment/healthcheck.sh

# Default command
CMD ["/app/deployment/entrypoint.sh"]

# Expose ports
EXPOSE 8080 8081 8082