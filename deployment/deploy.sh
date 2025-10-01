#!/bin/bash
set -euo pipefail

# PhysGrad Cloud Deployment Script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
NAMESPACE="physgrad"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-kubernetes}"

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

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."

    local missing_tools=()

    # Check for required tools
    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            missing_tools+=("kubectl")
        fi

        if ! command -v helm &> /dev/null; then
            log_warning "Helm not found - using direct kubectl deployment"
        fi
    fi

    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA CUDA support detected"
    else
        log_warning "NVIDIA CUDA not detected - will run in CPU mode"
    fi
}

# Function to build Docker image
build_image() {
    log_info "Building PhysGrad Docker image..."

    cd "$PROJECT_ROOT"

    # Build the image
    docker build -t "physgrad:${IMAGE_TAG}" -f Dockerfile .

    # Tag for registry if specified
    if [[ "$DOCKER_REGISTRY" != "localhost:5000" ]]; then
        docker tag "physgrad:${IMAGE_TAG}" "${DOCKER_REGISTRY}/physgrad:${IMAGE_TAG}"
        log_info "Pushing image to registry..."
        docker push "${DOCKER_REGISTRY}/physgrad:${IMAGE_TAG}"
    fi

    log_success "Docker image built successfully"
}

# Function to validate Kubernetes cluster
validate_k8s_cluster() {
    log_info "Validating Kubernetes cluster..."

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check for GPU node support
    local gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l)
    if [[ "$gpu_nodes" -eq 0 ]]; then
        log_warning "No GPU nodes found in cluster - will run in CPU mode"
    else
        log_success "Found $gpu_nodes GPU node(s) in cluster"
    fi

    # Check for required storage classes
    local storage_classes=("fast-ssd" "standard")
    for sc in "${storage_classes[@]}"; do
        if ! kubectl get storageclass "$sc" &> /dev/null; then
            log_warning "Storage class '$sc' not found - update PVC definitions if needed"
        fi
    done
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying PhysGrad to Kubernetes..."

    cd "$SCRIPT_DIR"

    # Create namespace
    kubectl apply -f k8s/namespace.yaml

    # Apply RBAC
    kubectl apply -f k8s/rbac.yaml

    # Apply ConfigMap
    kubectl apply -f k8s/configmap.yaml

    # Apply PVCs
    kubectl apply -f k8s/pvc.yaml

    # Wait for PVCs to be bound
    log_info "Waiting for persistent volumes to be provisioned..."
    kubectl wait --for=condition=Bound pvc/physgrad-data-pvc -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=Bound pvc/physgrad-results-pvc -n "$NAMESPACE" --timeout=300s

    # Apply deployment
    kubectl apply -f k8s/deployment.yaml

    # Apply services
    kubectl apply -f k8s/service.yaml

    # Apply HPA
    kubectl apply -f k8s/hpa.yaml

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available deployment/physgrad-compute -n "$NAMESPACE" --timeout=600s

    log_success "Kubernetes deployment completed successfully"
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying PhysGrad with Docker Compose..."

    cd "$SCRIPT_DIR"

    # Start services
    docker-compose up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose exec -T physgrad /app/deployment/healthcheck.sh &> /dev/null; then
            log_success "Services are healthy"
            break
        fi

        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services failed to become healthy within timeout"
            docker-compose logs
            exit 1
        fi

        echo -n "."
        sleep 10
        ((attempt++))
    done

    log_success "Docker Compose deployment completed successfully"
}

# Function to show deployment status
show_status() {
    log_info "Deployment Status:"
    echo "=================="

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        echo ""
        echo "Namespace: $NAMESPACE"
        echo ""
        echo "Pods:"
        kubectl get pods -n "$NAMESPACE" -o wide

        echo ""
        echo "Services:"
        kubectl get services -n "$NAMESPACE"

        echo ""
        echo "Ingress (if any):"
        kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "No ingress resources found"

        echo ""
        echo "Storage:"
        kubectl get pvc -n "$NAMESPACE"

        echo ""
        echo "Resource Usage:"
        kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics not available"

    else
        echo ""
        echo "Docker Compose Services:"
        docker-compose ps

        echo ""
        echo "Container Logs (last 10 lines):"
        docker-compose logs --tail=10
    fi
}

# Function to clean up deployment
cleanup() {
    log_info "Cleaning up PhysGrad deployment..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
        log_success "Kubernetes resources cleaned up"
    else
        cd "$SCRIPT_DIR"
        docker-compose down -v
        log_success "Docker Compose resources cleaned up"
    fi
}

# Function to show help
show_help() {
    cat << EOF
PhysGrad Cloud Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
  deploy    Deploy PhysGrad to the target environment
  status    Show deployment status
  cleanup   Remove all deployed resources
  build     Build Docker image only
  help      Show this help message

Options:
  --mode MODE           Deployment mode: kubernetes|docker-compose (default: kubernetes)
  --registry REGISTRY   Docker registry for image storage (default: localhost:5000)
  --tag TAG            Docker image tag (default: latest)
  --namespace NS       Kubernetes namespace (default: physgrad)

Environment Variables:
  DEPLOYMENT_MODE      Same as --mode
  DOCKER_REGISTRY      Same as --registry
  IMAGE_TAG           Same as --tag

Examples:
  $0 deploy                                    # Deploy to Kubernetes
  $0 --mode docker-compose deploy             # Deploy with Docker Compose
  $0 --registry gcr.io/my-project deploy      # Deploy with custom registry
  $0 status                                   # Show deployment status
  $0 cleanup                                  # Clean up all resources

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        deploy|status|cleanup|build|help)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate deployment mode
if [[ "$DEPLOYMENT_MODE" != "kubernetes" && "$DEPLOYMENT_MODE" != "docker-compose" ]]; then
    log_error "Invalid deployment mode: $DEPLOYMENT_MODE"
    exit 1
fi

# Main execution
main() {
    case "${COMMAND:-deploy}" in
        deploy)
            check_prerequisites
            build_image
            if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
                validate_k8s_cluster
                deploy_kubernetes
            else
                deploy_docker_compose
            fi
            show_status
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup
            ;;
        build)
            check_prerequisites
            build_image
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Unknown command: ${COMMAND}"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function
main