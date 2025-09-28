NVCC = nvcc
CXX = g++
CUDA_PATH = /usr/local/cuda

NVCC_FLAGS = -std=c++17 --use_fast_math -lineinfo --extended-lambda -arch=sm_75
CXX_FLAGS = -std=c++17 -O3 -Wall
INCLUDES = -I$(CUDA_PATH)/include -I./src
LIBS = -L$(CUDA_PATH)/lib64 -lcudart

SOURCES_CUDA = src/simulation.cu src/stability_improvements.cu src/multi_gpu.cu src/domain_decomposition.cu src/gpu_communication.cu src/load_balancer.cu
SOURCES_CPP = src/visualization.cpp src/collision_detection.cpp src/constraints.cpp src/differentiable_contact.cpp src/rigid_body.cpp src/symplectic_integrators.cpp
SOURCES_TESTS = tests/run_tests.cpp tests/test_physics.cpp tests/test_differentiable_contact.cpp tests/test_rigid_body.cpp tests/test_symplectic_integrators.cpp tests/test_constraints.cpp tests/test_collision_detection.cpp

OBJECTS_CUDA = $(SOURCES_CUDA:.cu=.o)
OBJECTS_CPP = $(SOURCES_CPP:.cpp=.o)
OBJECTS_TESTS = $(SOURCES_TESTS:.cpp=.o) tests/test_cuda.o

.PHONY: all clean test python

all: test

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

test: $(OBJECTS_CUDA) $(OBJECTS_CPP) $(OBJECTS_TESTS)
	$(NVCC) $(NVCC_FLAGS) -o run_tests $^ $(LIBS)

python:
	cd python && python setup.py build_ext --inplace

clean:
	rm -f src/*.o tests/*.o run_tests
	rm -rf python/build python/dist python/*.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

install-python:
	cd python && pip install -e .

.SECONDARY: $(OBJECTS_CUDA) $(OBJECTS_CPP) $(OBJECTS_TESTS)