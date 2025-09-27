NVCC = nvcc
CXX = g++
CUDA_PATH = /usr/local/cuda-12.1

NVCC_FLAGS = -std=c++17 --use_fast_math -lineinfo --extended-lambda -arch=sm_75
CXX_FLAGS = -std=c++17 -O3 -Wall
INCLUDES = -I$(CUDA_PATH)/include -I./src
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lm -lGL -lglfw

SOURCES_CUDA = src/simulation.cu src/stability_improvements.cu
SOURCES_CPP = src/main.cpp
SOURCES_CONSOLE = src/console.cpp
SOURCES_INVERSE = src/inverse_demo.cpp
SOURCES_OPTIMIZED_TRAJ = src/optimized_trajectory.cpp
SOURCES_ADAPTIVE_DEMO = src/adaptive_integration_demo.cpp
SOURCES_TAPE_DEMO = src/tape_compression_demo.cpp
SOURCES_CONFIG_DEMO = src/config_demo.cpp
SOURCES_LOGGING_DEMO = src/logging_demo.cpp
SOURCES_BATCHED_DEMO = src/batched_memory_demo.cpp

OBJECTS = $(SOURCES_CUDA:.cu=.o) $(SOURCES_CPP:.cpp=.o)
OBJECTS_CONSOLE = $(SOURCES_CUDA:.cu=.o) $(SOURCES_CONSOLE:.cpp=.o)
OBJECTS_INVERSE = $(SOURCES_CUDA:.cu=.o) $(SOURCES_INVERSE:.cpp=.o)
OBJECTS_OPTIMIZED_TRAJ = $(SOURCES_CUDA:.cu=.o) $(SOURCES_OPTIMIZED_TRAJ:.cpp=.o)
OBJECTS_ADAPTIVE_DEMO = $(SOURCES_ADAPTIVE_DEMO:.cpp=.o)
OBJECTS_TAPE_DEMO = $(SOURCES_TAPE_DEMO:.cpp=.o)
OBJECTS_CONFIG_DEMO = $(SOURCES_CONFIG_DEMO:.cpp=.o)
OBJECTS_LOGGING_DEMO = $(SOURCES_LOGGING_DEMO:.cpp=.o)
OBJECTS_BATCHED_DEMO = $(SOURCES_CUDA:.cu=.o) $(SOURCES_BATCHED_DEMO:.cpp=.o)

all: physgrad

physgrad: $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

console: $(OBJECTS_CONSOLE)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_console $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

inverse: $(OBJECTS_INVERSE)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_inverse $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

optimized_traj: $(OBJECTS_OPTIMIZED_TRAJ)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_optimized_traj $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

adaptive_demo: $(OBJECTS_ADAPTIVE_DEMO)
	$(CXX) $(CXX_FLAGS) -o physgrad_adaptive_demo $^ -lm

tape_demo: $(OBJECTS_TAPE_DEMO)
	$(CXX) $(CXX_FLAGS) -o physgrad_tape_demo $^ -lm

config_demo: $(OBJECTS_CONFIG_DEMO)
	$(CXX) $(CXX_FLAGS) -o physgrad_config_demo $^ -lm

logging_demo: $(OBJECTS_LOGGING_DEMO)
	$(CXX) $(CXX_FLAGS) -o physgrad_logging_demo $^ -lm

batched_demo: $(OBJECTS_BATCHED_DEMO)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_batched_demo $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

demos: adaptive_demo tape_demo config_demo logging_demo batched_demo optimized_traj

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

tests: console
	cd tests && $(MAKE)

clean:
	rm -f src/*.o physgrad physgrad_console physgrad_inverse physgrad_optimized_traj physgrad_adaptive_demo physgrad_tape_demo physgrad_config_demo physgrad_logging_demo physgrad_batched_demo

.PHONY: all console inverse optimized_traj adaptive_demo tape_demo config_demo logging_demo batched_demo demos tests clean