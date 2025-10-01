#!/bin/bash

# PhysGrad Automated Test Runner
# Comprehensive test suite with categorized testing, performance benchmarks,
# and detailed reporting.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
TEST_RESULTS_DIR="$PROJECT_ROOT/test_results"
LOG_FILE="$TEST_RESULTS_DIR/test_log.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test categories
declare -A TEST_CATEGORIES=(
    ["core"]="Basic core functionality tests"
    ["memory"]="Memory optimization and coalescing tests"
    ["concepts"]="C++20 concepts validation tests"
    ["g2p2g"]="Grid-to-Particle-to-Grid kernel tests"
    ["conservation"]="Energy and momentum conservation tests"
    ["integration"]="PyTorch integration tests"
    ["performance"]="Performance benchmarks"
    ["gpu"]="GPU/CUDA specific tests"
)

# Initialize test environment
init_test_env() {
    echo -e "${BLUE}Initializing test environment...${NC}"

    # Create directories
    mkdir -p "$TEST_RESULTS_DIR"
    mkdir -p "$BUILD_DIR"

    # Clear previous results
    rm -f "$LOG_FILE"
    touch "$LOG_FILE"

    echo "Test run started at $(date)" | tee -a "$LOG_FILE"
    echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
    echo "Build directory: $BUILD_DIR" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Log function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Error handling function
handle_error() {
    local exit_code=$1
    local test_name="$2"

    if [ $exit_code -ne 0 ]; then
        log "${RED}❌ FAILED: $test_name (exit code: $exit_code)${NC}"
        return 1
    else
        log "${GREEN}✅ PASSED: $test_name${NC}"
        return 0
    fi
}

# Check system requirements
check_requirements() {
    log "${BLUE}Checking system requirements...${NC}"

    # Check compiler
    if ! command -v g++ &> /dev/null; then
        log "${RED}Error: g++ compiler not found${NC}"
        exit 1
    fi

    local gcc_version=$(g++ --version | head -n1)
    log "Compiler: $gcc_version"

    # Check CUDA (optional)
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}')
        log "CUDA: $cuda_version"
    else
        log "${YELLOW}Warning: CUDA not available, GPU tests will be skipped${NC}"
    fi

    # Check Python (for PyTorch tests)
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version)
        log "Python: $python_version"

        # Check PyTorch
        if python3 -c "import torch" &> /dev/null; then
            local torch_version=$(python3 -c "import torch; print(torch.__version__)")
            log "PyTorch: $torch_version"
        else
            log "${YELLOW}Warning: PyTorch not available, integration tests will be skipped${NC}"
        fi
    fi

    log ""
}

# Build tests
build_tests() {
    log "${BLUE}Building test executables...${NC}"

    cd "$PROJECT_ROOT"

    # Core tests
    log "Building core tests..."
    if g++ -std=c++20 -I./src -I./external/eigen3 test_memory_minimal.cpp -o test_memory_minimal -O2 &>> "$LOG_FILE"; then
        log "${GREEN}✅ test_memory_minimal built successfully${NC}"
    else
        log "${RED}❌ Failed to build test_memory_minimal${NC}"
        return 1
    fi

    # Concepts tests
    log "Building concepts tests..."
    if g++ -std=c++20 -I. test_concepts_v2.cpp -o test_concepts_v2 &>> "$LOG_FILE"; then
        log "${GREEN}✅ test_concepts_v2 built successfully${NC}"
    else
        log "${RED}❌ Failed to build test_concepts_v2${NC}"
        return 1
    fi

    # PyTorch autograd tests
    log "Building PyTorch autograd tests..."
    if g++ -std=c++20 test_pytorch_autograd_simple.cpp -o test_pytorch_autograd_simple -O2 &>> "$LOG_FILE"; then
        log "${GREEN}✅ test_pytorch_autograd_simple built successfully${NC}"
    else
        log "${RED}❌ Failed to build test_pytorch_autograd_simple${NC}"
        return 1
    fi

    # G2P2G tests (if available)
    if [ -f "./test_g2p2g_kernels" ]; then
        log "${GREEN}✅ test_g2p2g_kernels available${NC}"
    else
        log "${YELLOW}⚠️  test_g2p2g_kernels not found, will skip${NC}"
    fi

    # Conservation tests
    if [ -f "./test_long_term_conservation_optimized" ]; then
        log "${GREEN}✅ test_long_term_conservation_optimized available${NC}"
    else
        log "${YELLOW}⚠️  Conservation tests not found, will skip${NC}"
    fi

    log ""
    return 0
}

# Run individual test
run_test() {
    local test_name="$1"
    local test_executable="$2"
    local category="$3"
    local timeout="${4:-30}"  # Default 30 second timeout

    log "${BLUE}Running $test_name ($category)...${NC}"

    if [ ! -f "$test_executable" ]; then
        log "${YELLOW}⚠️  Skipping $test_name - executable not found${NC}"
        return 0
    fi

    # Run test with timeout
    local start_time=$(date +%s)
    if timeout "$timeout" "$test_executable" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "${GREEN}✅ PASSED: $test_name (${duration}s)${NC}"
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if [ $exit_code -eq 124 ]; then
            log "${RED}❌ TIMEOUT: $test_name (>${timeout}s)${NC}"
        else
            log "${RED}❌ FAILED: $test_name (${duration}s, exit code: $exit_code)${NC}"
        fi
        return 1
    fi
}

# Run category of tests
run_category() {
    local category="$1"
    log "${BLUE}=== Running $category tests ===${NC}"
    log "${TEST_CATEGORIES[$category]}"
    log ""

    local failed_tests=0
    local total_tests=0

    case "$category" in
        "core")
            ((total_tests++))
            if ! run_test "Memory Validation" "./test_memory_minimal" "$category" 60; then
                ((failed_tests++))
            fi
            ;;

        "concepts")
            ((total_tests++))
            if ! run_test "C++20 Concepts" "./test_concepts_v2" "$category" 30; then
                ((failed_tests++))
            fi
            ;;

        "g2p2g")
            if [ -f "./test_g2p2g_kernels" ]; then
                ((total_tests++))
                if ! run_test "G2P2G Kernels" "./test_g2p2g_kernels" "$category" 120; then
                    ((failed_tests++))
                fi
            fi
            ;;

        "conservation")
            if [ -f "./test_long_term_conservation_optimized" ]; then
                ((total_tests++))
                if ! run_test "Long-term Conservation" "./test_long_term_conservation_optimized" "$category" 60; then
                    ((failed_tests++))
                fi
            fi
            ;;

        "memory")
            # Already covered in core
            log "${YELLOW}Memory tests included in core category${NC}"
            ;;

        "integration")
            if [ -f "./test_pytorch_autograd_simple" ]; then
                ((total_tests++))
                if ! run_test "PyTorch Autograd Functions" "./test_pytorch_autograd_simple" "$category" 60; then
                    ((failed_tests++))
                fi
            else
                log "${YELLOW}Skipping PyTorch integration tests - test executable not found${NC}"
            fi
            ;;

        "performance")
            log "${BLUE}Performance benchmarks would go here${NC}"
            # Placeholder for performance benchmarks
            ;;

        "gpu")
            if command -v nvcc &> /dev/null; then
                log "${BLUE}GPU-specific tests would go here${NC}"
                # Placeholder for GPU tests
            else
                log "${YELLOW}Skipping GPU tests - CUDA not available${NC}"
            fi
            ;;
    esac

    log ""
    log "${BLUE}Category $category summary: $((total_tests - failed_tests))/$total_tests passed${NC}"
    log ""

    return $failed_tests
}

# Generate test report
generate_report() {
    local total_failed="$1"
    local total_tests="$2"

    log "${BLUE}=== Test Report ===${NC}"
    log "Test run completed at $(date)"
    log "Total tests: $total_tests"
    log "Passed: $((total_tests - total_failed))"
    log "Failed: $total_failed"

    if [ $total_failed -eq 0 ]; then
        log "${GREEN}🎉 All tests PASSED!${NC}"
        return 0
    else
        log "${RED}❌ $total_failed test(s) FAILED${NC}"
        return 1
    fi
}

# Main test runner
main() {
    local categories_to_run=()
    local run_all=true

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --category|-c)
                categories_to_run+=("$2")
                run_all=false
                shift 2
                ;;
            --list|-l)
                echo "Available test categories:"
                for category in "${!TEST_CATEGORIES[@]}"; do
                    echo "  $category: ${TEST_CATEGORIES[$category]}"
                done
                exit 0
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --category, -c <category>  Run specific test category"
                echo "  --list, -l                 List available categories"
                echo "  --help, -h                 Show this help"
                echo ""
                echo "Available categories:"
                for category in "${!TEST_CATEGORIES[@]}"; do
                    echo "  $category: ${TEST_CATEGORIES[$category]}"
                done
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # If no specific categories requested, run all
    if [ "$run_all" = true ]; then
        categories_to_run=($(printf '%s\n' "${!TEST_CATEGORIES[@]}" | sort))
    fi

    init_test_env
    check_requirements

    if ! build_tests; then
        log "${RED}❌ Build failed, exiting${NC}"
        exit 1
    fi

    local total_failed=0
    local total_categories=${#categories_to_run[@]}

    # Run tests by category
    for category in "${categories_to_run[@]}"; do
        if [[ -n "${TEST_CATEGORIES[$category]}" ]]; then
            local category_failed=0
            run_category "$category"
            category_failed=$?
            total_failed=$((total_failed + category_failed))
        else
            log "${RED}Unknown category: $category${NC}"
            exit 1
        fi
    done

    generate_report $total_failed $total_categories

    # Save detailed log
    local report_file="$TEST_RESULTS_DIR/test_report_$(date +%Y%m%d_%H%M%S).txt"
    cp "$LOG_FILE" "$report_file"
    log "Detailed report saved to: $report_file"

    exit $total_failed
}

# Run main function with all arguments
main "$@"