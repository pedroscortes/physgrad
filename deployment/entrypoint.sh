#!/bin/bash
set -euo pipefail

# PhysGrad Cloud Deployment Entrypoint
echo "Starting PhysGrad Cloud Service..."

# Environment validation
validate_environment() {
    local required_vars=("PHYSICS_TIMESTEP" "MAX_PARTICLES" "API_PORT")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            echo "ERROR: Required environment variable $var is not set"
            exit 1
        fi
    done
}

# CUDA device detection and validation
setup_cuda() {
    echo "Detecting CUDA devices..."
    if ! nvidia-smi &>/dev/null; then
        echo "WARNING: nvidia-smi not available, running in CPU mode"
        export ENABLE_CUDA=false
        return
    fi

    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $gpu_count GPU(s)"

    if [[ "$gpu_count" -eq 0 ]]; then
        echo "WARNING: No GPUs detected, running in CPU mode"
        export ENABLE_CUDA=false
    else
        export ENABLE_CUDA=true
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
        echo "Using GPU device(s): $CUDA_VISIBLE_DEVICES"
    fi
}

# Initialize data directories
setup_directories() {
    local dirs=("${DATA_PATH}" "${RESULTS_PATH}" "${LOGS_PATH}")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            echo "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done

    # Verify write permissions
    for dir in "${dirs[@]}"; do
        if [[ ! -w "$dir" ]]; then
            echo "ERROR: Cannot write to directory: $dir"
            exit 1
        fi
    done
}

# Start background services
start_services() {
    echo "Starting PhysGrad services..."

    # Start metrics server
    if [[ "${METRICS_ENABLED:-true}" == "true" ]]; then
        echo "Starting metrics server on port ${METRICS_PORT}"
        /app/bin/physgrad_metrics_server \
            --port="${METRICS_PORT}" \
            --log-level="${LOG_LEVEL}" &
        METRICS_PID=$!
        echo "Metrics server started with PID: $METRICS_PID"
    fi

    # Start health check server
    echo "Starting health check server on port ${HEALTH_PORT}"
    /app/bin/physgrad_health_server \
        --port="${HEALTH_PORT}" \
        --data-path="${DATA_PATH}" &
    HEALTH_PID=$!
    echo "Health server started with PID: $HEALTH_PID"

    # Wait for services to be ready
    sleep 5

    # Verify services are responding
    if ! curl -f "http://localhost:${HEALTH_PORT}/health" &>/dev/null; then
        echo "ERROR: Health server not responding"
        exit 1
    fi

    if [[ "${METRICS_ENABLED:-true}" == "true" ]] && ! curl -f "http://localhost:${METRICS_PORT}/metrics" &>/dev/null; then
        echo "WARNING: Metrics server not responding"
    fi
}

# Cleanup function for graceful shutdown
cleanup() {
    echo "Shutting down PhysGrad services..."

    # Kill background processes
    if [[ -n "${METRICS_PID:-}" ]]; then
        kill "$METRICS_PID" 2>/dev/null || true
    fi

    if [[ -n "${HEALTH_PID:-}" ]]; then
        kill "$HEALTH_PID" 2>/dev/null || true
    fi

    # Wait for processes to exit
    wait 2>/dev/null || true

    echo "Cleanup completed"
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    echo "PhysGrad Cloud Service v1.0.0"
    echo "Node: ${NODE_NAME:-unknown}"
    echo "Pod: ${POD_NAME:-unknown}"
    echo "Namespace: ${POD_NAMESPACE:-default}"

    validate_environment
    setup_cuda
    setup_directories
    start_services

    echo "Starting main PhysGrad API server on port ${API_PORT}"

    # Start the main application
    exec /app/bin/physgrad_server \
        --port="${API_PORT}" \
        --data-path="${DATA_PATH}" \
        --results-path="${RESULTS_PATH}" \
        --log-path="${LOGS_PATH}" \
        --log-level="${LOG_LEVEL}" \
        --log-format="${LOG_FORMAT}" \
        --max-particles="${MAX_PARTICLES}" \
        --timestep="${PHYSICS_TIMESTEP}" \
        --threads-per-block="${THREADS_PER_BLOCK}" \
        --enable-cuda="${ENABLE_CUDA:-true}"
}

# Execute main function
main "$@"