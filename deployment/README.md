# PhysGrad Cloud Deployment

This directory contains all the necessary files and configurations for deploying PhysGrad in cloud-native environments including Kubernetes and Docker Compose.

## Overview

PhysGrad is a high-performance physics simulation framework that supports:
- CUDA-accelerated computing
- Material Point Method (MPM) simulations
- Quantum-classical hybrid simulations
- Real-time physics visualization
- Cloud-native scalability

## Deployment Options

### 1. Kubernetes Deployment (Recommended for Production)

```bash
# Deploy to Kubernetes cluster
./deploy.sh deploy

# Deploy with custom settings
./deploy.sh --registry gcr.io/my-project --tag v1.0.0 deploy

# Check deployment status
./deploy.sh status

# Clean up deployment
./deploy.sh cleanup
```

### 2. Docker Compose Deployment (Development/Testing)

```bash
# Deploy with Docker Compose
./deploy.sh --mode docker-compose deploy

# View logs
docker-compose logs -f physgrad

# Stop services
docker-compose down
```

## Prerequisites

### Required Tools
- **Docker**: Container runtime and image building
- **kubectl**: Kubernetes command-line tool (for K8s deployment)
- **docker-compose**: Multi-container Docker applications (for local deployment)

### Optional Tools
- **helm**: Kubernetes package manager (recommended)
- **nvidia-docker**: GPU support in containers

### System Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM
- **GPU**: NVIDIA GPU with CUDA support (optional, will fallback to CPU)
- **Storage**: 100GB+ available disk space

## Architecture

### Container Structure
```
physgrad:latest
├── /app/bin/           # PhysGrad executables
├── /app/lib/           # Shared libraries
├── /app/src/           # Source code (for reference)
├── /app/data/          # Input data storage
├── /app/results/       # Simulation results
├── /app/logs/          # Application logs
└── /app/deployment/    # Deployment scripts
```

### Kubernetes Architecture
```
Namespace: physgrad
├── Deployment: physgrad-compute (2+ replicas)
├── Services: API, Metrics, Health endpoints
├── ConfigMap: Application configuration
├── PVC: Persistent storage for data and results
├── HPA: Horizontal Pod Autoscaler
└── RBAC: Service account and permissions
```

### Monitoring Stack
```
├── Prometheus: Metrics collection and alerting
├── Grafana: Visualization and dashboards
└── Custom Metrics: PhysGrad-specific performance metrics
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PHYSICS_TIMESTEP` | `0.001` | Simulation timestep |
| `MAX_PARTICLES` | `10000000` | Maximum particle count |
| `CUDA_DEVICE_COUNT` | `auto` | Number of CUDA devices |
| `THREADS_PER_BLOCK` | `256` | CUDA threads per block |
| `MEMORY_POOL_SIZE` | `2GB` | GPU memory pool size |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `METRICS_ENABLED` | `true` | Enable metrics collection |
| `API_PORT` | `8080` | Main API server port |
| `METRICS_PORT` | `8081` | Metrics endpoint port |
| `HEALTH_PORT` | `8082` | Health check port |

### Storage Configuration

#### Kubernetes Persistent Volumes
- **Data Volume**: 100GB (fast-ssd) - Input datasets and simulation parameters
- **Results Volume**: 500GB (standard) - Simulation outputs and visualizations

#### Docker Compose Volumes
- **physgrad_data**: Input data storage
- **physgrad_results**: Output results storage
- **physgrad_logs**: Application logs

## API Endpoints

### Main API (Port 8080)
- `GET /status` - Service status
- `POST /simulation` - Start simulation
- `GET /simulation/{id}` - Get simulation status
- `DELETE /simulation/{id}` - Stop simulation

### Health Check API (Port 8082)
- `GET /health` - Basic health check
- `GET /ready` - Readiness probe
- `GET /health/detailed` - Extended health information

### Metrics API (Port 8081)
- `GET /metrics` - Prometheus metrics

## Security

### Container Security
- **Non-root user**: Runs as user `physgrad` (UID 1000)
- **Read-only root filesystem**: Prevents runtime modifications
- **Minimal attack surface**: Only necessary capabilities
- **No privilege escalation**: Security constraints enforced

### Network Security
- **Internal communication**: Services communicate within cluster network
- **TLS encryption**: HTTPS endpoints (when configured)
- **Access control**: RBAC policies for Kubernetes resources

## Monitoring and Alerting

### Available Metrics
- **Performance**: Simulation FPS, particles per second
- **Resource Usage**: CPU, memory, GPU utilization
- **API Metrics**: Request rate, latency, error rate
- **System Health**: Service status, disk space, temperature

### Alert Rules
- **Service Down**: API service unavailable
- **High Error Rate**: Excessive API errors
- **Resource Exhaustion**: High CPU/memory/GPU usage
- **Storage Issues**: Low disk space
- **Performance Degradation**: Slow simulations

### Grafana Dashboards
- **Overview**: Service status and key metrics
- **Performance**: Simulation performance metrics
- **Infrastructure**: Resource utilization
- **API Monitoring**: Request patterns and errors

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check container GPU access
docker run --gpus all physgrad:latest nvidia-smi
```

#### 2. Memory Issues
```bash
# Check memory usage
kubectl top pods -n physgrad

# Scale down if needed
kubectl scale deployment physgrad-compute --replicas=1 -n physgrad
```

#### 3. Storage Problems
```bash
# Check PVC status
kubectl get pvc -n physgrad

# Check available disk space
kubectl exec -it deployment/physgrad-compute -n physgrad -- df -h
```

### Debug Commands

#### Kubernetes Debugging
```bash
# View pod logs
kubectl logs deployment/physgrad-compute -n physgrad

# Execute into pod
kubectl exec -it deployment/physgrad-compute -n physgrad -- bash

# Check resource usage
kubectl describe pod -n physgrad

# View events
kubectl get events -n physgrad --sort-by='.lastTimestamp'
```

#### Docker Compose Debugging
```bash
# View logs
docker-compose logs physgrad

# Execute into container
docker-compose exec physgrad bash

# Check container stats
docker stats
```

### Performance Tuning

#### GPU Optimization
```bash
# Monitor GPU utilization
watch nvidia-smi

# Adjust CUDA settings
export CUDA_VISIBLE_DEVICES=0,1
export THREADS_PER_BLOCK=512
```

#### Memory Optimization
```bash
# Reduce particle count for testing
export MAX_PARTICLES=1000000

# Adjust memory pool size
export MEMORY_POOL_SIZE=1GB
```

## Validation

### Pre-deployment Validation
```bash
# Validate all components
./validate.sh

# Validate specific components
./validate.sh --k8s-only
./validate.sh --docker-only
./validate.sh --scripts-only
```

### Post-deployment Testing
```bash
# Health check
curl http://localhost:8082/health

# API status
curl http://localhost:8080/status

# Metrics endpoint
curl http://localhost:8081/metrics
```

## Scaling

### Horizontal Scaling
```bash
# Scale up replicas
kubectl scale deployment physgrad-compute --replicas=5 -n physgrad

# Auto-scaling is configured via HPA:
# - CPU utilization > 70%
# - Memory utilization > 80%
# - Active simulations > 5 per pod
```

### Vertical Scaling
```bash
# Update resource limits in deployment.yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: 2
  limits:
    memory: "32Gi"
    cpu: "16"
    nvidia.com/gpu: 2
```

## Development

### Building Custom Images
```bash
# Build with custom tag
./deploy.sh --tag dev-v1.0 build

# Push to registry
./deploy.sh --registry my-registry.com --tag dev-v1.0 build
```

### Local Development
```bash
# Run with local code changes
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Hot reload (requires volume mounts)
# See docker-compose.dev.yml for development overrides
```

## Support

### Log Locations
- **Kubernetes**: `kubectl logs deployment/physgrad-compute -n physgrad`
- **Docker Compose**: `docker-compose logs physgrad`
- **Container logs**: `/app/logs/`

### Configuration Files
- **Kubernetes**: `k8s/*.yaml`
- **Docker Compose**: `docker-compose.yml`
- **Monitoring**: `monitoring/*.yml`

### Validation and Testing
- **Validation script**: `./validate.sh`
- **Health checks**: `./healthcheck.sh`
- **Deployment script**: `./deploy.sh help`

For additional support, refer to the main PhysGrad documentation or create an issue in the project repository.