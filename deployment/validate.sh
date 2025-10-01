#!/bin/bash
set -euo pipefail

# PhysGrad Cloud Deployment Validation Script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to validate Kubernetes manifests
validate_k8s_manifests() {
    log_info "Validating Kubernetes manifests..."

    local manifests_dir="$SCRIPT_DIR/k8s"
    local errors=0

    # Check if manifest files exist
    local required_files=(
        "namespace.yaml"
        "configmap.yaml"
        "deployment.yaml"
        "service.yaml"
        "pvc.yaml"
        "rbac.yaml"
        "hpa.yaml"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$manifests_dir/$file" ]]; then
            log_error "Missing manifest file: $file"
            ((errors++))
        else
            # Validate YAML syntax
            if ! kubectl apply --dry-run=client -f "$manifests_dir/$file" &>/dev/null; then
                log_error "Invalid YAML syntax in $file"
                ((errors++))
            else
                log_success "Valid manifest: $file"
            fi
        fi
    done

    return $errors
}

# Function to validate Docker configuration
validate_docker_config() {
    log_info "Validating Docker configuration..."

    local errors=0

    # Check Dockerfile
    if [[ ! -f "$SCRIPT_DIR/../Dockerfile" ]]; then
        log_error "Dockerfile not found"
        ((errors++))
    else
        log_success "Dockerfile found"
    fi

    # Check docker-compose.yml
    if [[ ! -f "$SCRIPT_DIR/docker-compose.yml" ]]; then
        log_error "docker-compose.yml not found"
        ((errors++))
    else
        # Validate docker-compose syntax
        if ! docker-compose -f "$SCRIPT_DIR/docker-compose.yml" config &>/dev/null; then
            log_error "Invalid docker-compose.yml syntax"
            ((errors++))
        else
            log_success "Valid docker-compose.yml"
        fi
    fi

    return $errors
}

# Function to validate monitoring configuration
validate_monitoring_config() {
    log_info "Validating monitoring configuration..."

    local errors=0
    local monitoring_dir="$SCRIPT_DIR/monitoring"

    # Check Prometheus configuration
    if [[ ! -f "$monitoring_dir/prometheus.yml" ]]; then
        log_error "Prometheus configuration not found"
        ((errors++))
    else
        log_success "Prometheus configuration found"
    fi

    # Check alert rules
    if [[ ! -f "$monitoring_dir/alert_rules.yml" ]]; then
        log_error "Alert rules not found"
        ((errors++))
    else
        log_success "Alert rules found"
    fi

    # Check Grafana configuration
    if [[ ! -d "$monitoring_dir/grafana" ]]; then
        log_error "Grafana configuration directory not found"
        ((errors++))
    else
        log_success "Grafana configuration found"
    fi

    return $errors
}

# Function to validate deployment scripts
validate_scripts() {
    log_info "Validating deployment scripts..."

    local errors=0
    local scripts=(
        "deploy.sh"
        "entrypoint.sh"
        "healthcheck.sh"
    )

    for script in "${scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"

        if [[ ! -f "$script_path" ]]; then
            log_error "Script not found: $script"
            ((errors++))
        elif [[ ! -x "$script_path" ]]; then
            log_error "Script not executable: $script"
            ((errors++))
        else
            # Basic syntax check for bash scripts
            if ! bash -n "$script_path"; then
                log_error "Syntax error in script: $script"
                ((errors++))
            else
                log_success "Valid script: $script"
            fi
        fi
    done

    return $errors
}

# Function to validate environment requirements
validate_environment() {
    log_info "Validating environment requirements..."

    local warnings=0

    # Check for Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found - required for containerized deployment"
        ((warnings++))
    else
        log_success "Docker found"

        # Check Docker daemon
        if ! docker info &>/dev/null; then
            log_warning "Docker daemon not running"
            ((warnings++))
        else
            log_success "Docker daemon running"
        fi
    fi

    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl not found - required for Kubernetes deployment"
        ((warnings++))
    else
        log_success "kubectl found"
    fi

    # Check for docker-compose
    if ! command -v docker-compose &> /dev/null; then
        log_warning "docker-compose not found - required for local deployment"
        ((warnings++))
    else
        log_success "docker-compose found"
    fi

    # Check for NVIDIA GPU support
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU support detected"
    else
        log_warning "NVIDIA GPU support not detected - will run in CPU mode"
        ((warnings++))
    fi

    return $warnings
}

# Function to run comprehensive validation
run_validation() {
    log_info "Starting PhysGrad deployment validation..."
    echo "==========================================="

    local total_errors=0
    local total_warnings=0

    # Run all validation checks
    validate_environment
    total_warnings=$((total_warnings + $?))

    validate_scripts
    total_errors=$((total_errors + $?))

    validate_docker_config
    total_errors=$((total_errors + $?))

    if command -v kubectl &> /dev/null; then
        validate_k8s_manifests
        total_errors=$((total_errors + $?))
    else
        log_warning "Skipping Kubernetes validation - kubectl not available"
        ((total_warnings++))
    fi

    validate_monitoring_config
    total_errors=$((total_errors + $?))

    # Summary
    echo ""
    log_info "Validation Summary:"
    echo "=================="
    echo "Errors: $total_errors"
    echo "Warnings: $total_warnings"

    if [[ $total_errors -eq 0 ]]; then
        log_success "All validation checks passed!"
        if [[ $total_warnings -gt 0 ]]; then
            log_warning "There are $total_warnings warnings - review before production deployment"
        fi
        return 0
    else
        log_error "Validation failed with $total_errors errors"
        return 1
    fi
}

# Function to show validation help
show_help() {
    cat << EOF
PhysGrad Deployment Validation Script

Usage: $0 [OPTIONS]

Options:
  --k8s-only       Validate only Kubernetes manifests
  --docker-only    Validate only Docker configuration
  --scripts-only   Validate only deployment scripts
  --env-only       Validate only environment requirements
  --help           Show this help message

Description:
  This script validates all components of the PhysGrad cloud deployment:
  - Kubernetes manifests syntax and completeness
  - Docker configuration and compose files
  - Deployment script syntax and executability
  - Environment requirements and tool availability
  - Monitoring configuration files

Exit Codes:
  0 - All validations passed (warnings allowed)
  1 - Validation failed with errors

EOF
}

# Parse command line arguments
case "${1:-all}" in
    --k8s-only)
        validate_k8s_manifests
        exit $?
        ;;
    --docker-only)
        validate_docker_config
        exit $?
        ;;
    --scripts-only)
        validate_scripts
        exit $?
        ;;
    --env-only)
        validate_environment
        exit $?
        ;;
    --help|-h)
        show_help
        exit 0
        ;;
    all|*)
        run_validation
        exit $?
        ;;
esac