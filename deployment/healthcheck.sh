#!/bin/bash
set -euo pipefail

# PhysGrad Health Check Script
# Used by Docker/Kubernetes health probes

HEALTH_PORT=${HEALTH_PORT:-8082}
TIMEOUT=${HEALTH_TIMEOUT:-5}

# Function to check if a service is responding
check_service() {
    local url="$1"
    local name="$2"

    if curl -f -s -m "$TIMEOUT" "$url" >/dev/null 2>&1; then
        echo "✓ $name is healthy"
        return 0
    else
        echo "✗ $name is unhealthy"
        return 1
    fi
}

# Function to check GPU availability (if CUDA is enabled)
check_gpu() {
    if [[ "${ENABLE_CUDA:-true}" == "true" ]]; then
        if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits >/dev/null 2>&1; then
            echo "✓ GPU is accessible"
            return 0
        else
            echo "✗ GPU is not accessible"
            return 1
        fi
    else
        echo "ℹ GPU check skipped (CUDA disabled)"
        return 0
    fi
}

# Function to check disk space
check_disk_space() {
    local data_path="${DATA_PATH:-/app/data}"
    local results_path="${RESULTS_PATH:-/app/results}"

    # Check if data directory has at least 1GB free
    local data_free=$(df "$data_path" | awk 'NR==2 {print $4}')
    if [[ "$data_free" -lt 1048576 ]]; then  # 1GB in KB
        echo "✗ Insufficient disk space in data directory"
        return 1
    fi

    # Check if results directory has at least 5GB free
    local results_free=$(df "$results_path" | awk 'NR==2 {print $4}')
    if [[ "$results_free" -lt 5242880 ]]; then  # 5GB in KB
        echo "✗ Insufficient disk space in results directory"
        return 1
    fi

    echo "✓ Disk space is adequate"
    return 0
}

# Function to check memory usage
check_memory() {
    # Get memory usage percentage
    local mem_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')

    if [[ "$mem_usage" -gt 90 ]]; then
        echo "✗ Memory usage too high: ${mem_usage}%"
        return 1
    fi

    echo "✓ Memory usage is normal: ${mem_usage}%"
    return 0
}

# Function to check process health
check_processes() {
    # Check if main PhysGrad processes are running
    local processes=("physgrad_server" "physgrad_health_server")

    if [[ "${METRICS_ENABLED:-true}" == "true" ]]; then
        processes+=("physgrad_metrics_server")
    fi

    for process in "${processes[@]}"; do
        if pgrep "$process" >/dev/null; then
            echo "✓ $process is running"
        else
            echo "✗ $process is not running"
            return 1
        fi
    done

    return 0
}

# Main health check function
main() {
    local exit_code=0

    echo "PhysGrad Health Check - $(date)"
    echo "=================================="

    # Basic service health
    if ! check_service "http://localhost:${HEALTH_PORT}/health" "Health API"; then
        exit_code=1
    fi

    # Extended checks for detailed health endpoint
    if [[ "${1:-}" == "--detailed" ]] || [[ "${HEALTH_CHECK_DETAILED:-false}" == "true" ]]; then
        echo ""
        echo "Extended Health Checks:"
        echo "----------------------"

        if ! check_gpu; then
            exit_code=1
        fi

        if ! check_disk_space; then
            exit_code=1
        fi

        if ! check_memory; then
            exit_code=1
        fi

        if ! check_processes; then
            exit_code=1
        fi

        # Check API responsiveness
        if ! check_service "http://localhost:${API_PORT:-8080}/status" "Main API"; then
            exit_code=1
        fi

        # Check metrics endpoint (if enabled)
        if [[ "${METRICS_ENABLED:-true}" == "true" ]]; then
            if ! check_service "http://localhost:${METRICS_PORT:-8081}/metrics" "Metrics API"; then
                # Metrics failure is non-fatal
                echo "⚠ Metrics endpoint unhealthy (non-fatal)"
            fi
        fi
    fi

    echo ""
    if [[ "$exit_code" -eq 0 ]]; then
        echo "Overall Status: ✓ HEALTHY"
    else
        echo "Overall Status: ✗ UNHEALTHY"
    fi

    exit "$exit_code"
}

# Execute main function with all arguments
main "$@"