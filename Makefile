NVCC = nvcc
CXX = g++
CUDA_PATH = /usr/local/cuda-12.1

NVCC_FLAGS = -std=c++17 --use_fast_math -lineinfo --extended-lambda -arch=sm_75
CXX_FLAGS = -std=c++17 -O3 -Wall
INCLUDES = -I$(CUDA_PATH)/include -I./src -I./external/imgui -I./external/imgui/backends
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lm -lGL -lglfw -lGLEW
VIZ_LIBS = $(LIBS) -ldl -lX11 -lXi -lXrandr -lXinerama -lXcursor -pthread

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
SOURCES_VIZ_DEMO = src/visualization_demo.cpp
SOURCES_VIZ = src/visualization.cpp
SOURCES_COLLISION = src/collision_detection.cpp
SOURCES_COLLISION_DEMO = src/collision_demo.cpp
SOURCES_INTERACTIVE_DEMO = src/interactive_demo.cpp
SOURCES_CONSTRAINTS = src/constraints.cpp
SOURCES_CONSTRAINTS_DEMO = src/constraints_demo.cpp
SOURCES_IMGUI = external/imgui/imgui.cpp external/imgui/imgui_demo.cpp external/imgui/imgui_draw.cpp external/imgui/imgui_tables.cpp external/imgui/imgui_widgets.cpp external/imgui/backends/imgui_impl_glfw.cpp external/imgui/backends/imgui_impl_opengl3.cpp

OBJECTS = $(SOURCES_CUDA:.cu=.o) $(SOURCES_CPP:.cpp=.o)
OBJECTS_CONSOLE = $(SOURCES_CUDA:.cu=.o) $(SOURCES_CONSOLE:.cpp=.o)
OBJECTS_INVERSE = $(SOURCES_CUDA:.cu=.o) $(SOURCES_INVERSE:.cpp=.o)
OBJECTS_OPTIMIZED_TRAJ = $(SOURCES_CUDA:.cu=.o) $(SOURCES_OPTIMIZED_TRAJ:.cpp=.o)
OBJECTS_ADAPTIVE_DEMO = $(SOURCES_ADAPTIVE_DEMO:.cpp=.o)
OBJECTS_TAPE_DEMO = $(SOURCES_TAPE_DEMO:.cpp=.o)
OBJECTS_CONFIG_DEMO = $(SOURCES_CONFIG_DEMO:.cpp=.o)
OBJECTS_LOGGING_DEMO = $(SOURCES_LOGGING_DEMO:.cpp=.o)
OBJECTS_BATCHED_DEMO = $(SOURCES_CUDA:.cu=.o) $(SOURCES_BATCHED_DEMO:.cpp=.o)
OBJECTS_VIZ_DEMO = $(SOURCES_CUDA:.cu=.o) $(SOURCES_VIZ:.cpp=.o) $(SOURCES_VIZ_DEMO:.cpp=.o) $(SOURCES_IMGUI:.cpp=.o)
OBJECTS_COLLISION_DEMO = $(SOURCES_CUDA:.cu=.o) $(SOURCES_VIZ:.cpp=.o) $(SOURCES_COLLISION:.cpp=.o) $(SOURCES_COLLISION_DEMO:.cpp=.o) $(SOURCES_IMGUI:.cpp=.o)
OBJECTS_INTERACTIVE_DEMO = $(SOURCES_CUDA:.cu=.o) $(SOURCES_VIZ:.cpp=.o) $(SOURCES_COLLISION:.cpp=.o) $(SOURCES_INTERACTIVE_DEMO:.cpp=.o) $(SOURCES_IMGUI:.cpp=.o)
OBJECTS_CONSTRAINTS_DEMO = $(SOURCES_CUDA:.cu=.o) $(SOURCES_VIZ:.cpp=.o) $(SOURCES_COLLISION:.cpp=.o) $(SOURCES_CONSTRAINTS:.cpp=.o) $(SOURCES_CONSTRAINTS_DEMO:.cpp=.o) $(SOURCES_IMGUI:.cpp=.o)

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

viz_demo: $(OBJECTS_VIZ_DEMO)
	$(CXX) $(CXX_FLAGS) -o physgrad_viz_demo $^ $(VIZ_LIBS)

collision_demo: $(OBJECTS_COLLISION_DEMO)
	$(CXX) $(CXX_FLAGS) -o physgrad_collision_demo $^ $(VIZ_LIBS)

interactive_demo: $(OBJECTS_INTERACTIVE_DEMO)
	$(CXX) $(CXX_FLAGS) -o physgrad_interactive_demo $^ $(VIZ_LIBS)

constraints_demo: $(OBJECTS_CONSTRAINTS_DEMO)
	$(CXX) $(CXX_FLAGS) -o physgrad_constraints_demo $^ $(VIZ_LIBS)

demos: adaptive_demo tape_demo config_demo logging_demo batched_demo optimized_traj viz_demo collision_demo interactive_demo constraints_demo

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

tests: console
	cd tests && $(MAKE)

clean:
	rm -f src/*.o external/imgui/*.o external/imgui/backends/*.o physgrad physgrad_console physgrad_inverse physgrad_optimized_traj physgrad_adaptive_demo physgrad_tape_demo physgrad_config_demo physgrad_logging_demo physgrad_batched_demo physgrad_viz_demo physgrad_collision_demo physgrad_interactive_demo physgrad_constraints_demo

.PHONY: all console inverse optimized_traj adaptive_demo tape_demo config_demo logging_demo batched_demo viz_demo collision_demo interactive_demo constraints_demo demos tests clean