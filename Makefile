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
SOURCES_VALIDATION = src/gradient_validation.cpp
SOURCES_SIMPLE_TEST = src/simple_grad_test.cpp
SOURCES_STABILITY = src/stability_test.cpp
SOURCES_PARAM_GRAD = src/parameter_gradients_test.cpp
SOURCES_SIMPLE_PARAM = src/simple_param_test.cpp
SOURCES_PARAM_LEARNING = src/parameter_learning_demo.cpp
SOURCES_IMPROVED_LEARNING = src/improved_parameter_learning.cpp
SOURCES_TIME_GRAD_TEST = src/time_step_gradient_test.cpp
SOURCES_SIMPLE_DT_TEST = src/simple_dt_gradient_test.cpp
SOURCES_DEBUG_DT_FORCES = src/debug_dt_forces.cpp
SOURCES_DIRECT_DT_TEST = src/direct_dt_gradient_test.cpp
SOURCES_TRAJECTORY = src/trajectory_optimization.cpp
SOURCES_SIMPLE_TRAJ = src/simple_trajectory_test.cpp
SOURCES_GRAD_CHECK = src/gradient_check_trajectory.cpp
SOURCES_DEBUG_TAPE = src/debug_gradient_tape.cpp
OBJECTS = $(SOURCES_CUDA:.cu=.o) $(SOURCES_CPP:.cpp=.o)
OBJECTS_CONSOLE = $(SOURCES_CUDA:.cu=.o) $(SOURCES_CONSOLE:.cpp=.o)
OBJECTS_INVERSE = $(SOURCES_CUDA:.cu=.o) $(SOURCES_INVERSE:.cpp=.o)
OBJECTS_VALIDATION = $(SOURCES_CUDA:.cu=.o) $(SOURCES_VALIDATION:.cpp=.o)
OBJECTS_SIMPLE_TEST = $(SOURCES_CUDA:.cu=.o) $(SOURCES_SIMPLE_TEST:.cpp=.o)
OBJECTS_STABILITY = $(SOURCES_CUDA:.cu=.o) $(SOURCES_STABILITY:.cpp=.o)
OBJECTS_PARAM_GRAD = $(SOURCES_CUDA:.cu=.o) $(SOURCES_PARAM_GRAD:.cpp=.o)
OBJECTS_SIMPLE_PARAM = $(SOURCES_CUDA:.cu=.o) $(SOURCES_SIMPLE_PARAM:.cpp=.o)
OBJECTS_PARAM_LEARNING = $(SOURCES_CUDA:.cu=.o) $(SOURCES_PARAM_LEARNING:.cpp=.o)
OBJECTS_IMPROVED_LEARNING = $(SOURCES_CUDA:.cu=.o) $(SOURCES_IMPROVED_LEARNING:.cpp=.o)
OBJECTS_TIME_GRAD_TEST = $(SOURCES_CUDA:.cu=.o) $(SOURCES_TIME_GRAD_TEST:.cpp=.o)
OBJECTS_SIMPLE_DT_TEST = $(SOURCES_CUDA:.cu=.o) $(SOURCES_SIMPLE_DT_TEST:.cpp=.o)
OBJECTS_DEBUG_DT_FORCES = $(SOURCES_CUDA:.cu=.o) $(SOURCES_DEBUG_DT_FORCES:.cpp=.o)
OBJECTS_DIRECT_DT_TEST = $(SOURCES_CUDA:.cu=.o) $(SOURCES_DIRECT_DT_TEST:.cpp=.o)
OBJECTS_TRAJECTORY = $(SOURCES_CUDA:.cu=.o) $(SOURCES_TRAJECTORY:.cpp=.o)
OBJECTS_SIMPLE_TRAJ = $(SOURCES_CUDA:.cu=.o) $(SOURCES_SIMPLE_TRAJ:.cpp=.o)
OBJECTS_GRAD_CHECK = $(SOURCES_CUDA:.cu=.o) $(SOURCES_GRAD_CHECK:.cpp=.o)
OBJECTS_DEBUG_TAPE = $(SOURCES_CUDA:.cu=.o) $(SOURCES_DEBUG_TAPE:.cpp=.o)

all: physgrad

physgrad: $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

console: $(OBJECTS_CONSOLE)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_console $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

inverse: $(OBJECTS_INVERSE)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_inverse $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

validation: $(OBJECTS_VALIDATION)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_validation $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

simple_test: $(OBJECTS_SIMPLE_TEST)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_simple_test $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

stability: $(OBJECTS_STABILITY)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_stability $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

param_grad: $(OBJECTS_PARAM_GRAD)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_param_grad $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

simple_param: $(OBJECTS_SIMPLE_PARAM)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_simple_param $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

param_learning: $(OBJECTS_PARAM_LEARNING)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_param_learning $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

improved_learning: $(OBJECTS_IMPROVED_LEARNING)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_improved_learning $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

time_grad_test: $(OBJECTS_TIME_GRAD_TEST)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_time_grad_test $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

simple_dt_test: $(OBJECTS_SIMPLE_DT_TEST)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_simple_dt_test $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

debug_dt_forces: $(OBJECTS_DEBUG_DT_FORCES)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_debug_dt_forces $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

direct_dt_test: $(OBJECTS_DIRECT_DT_TEST)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_direct_dt_test $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

trajectory: $(OBJECTS_TRAJECTORY)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_trajectory $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

simple_traj: $(OBJECTS_SIMPLE_TRAJ)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_simple_traj $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

grad_check: $(OBJECTS_GRAD_CHECK)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_grad_check $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

debug_tape: $(OBJECTS_DEBUG_TAPE)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_debug_tape $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

tests: console
	cd tests && $(MAKE)

clean:
	rm -f src/*.o physgrad physgrad_console physgrad_inverse physgrad_validation physgrad_simple_test physgrad_stability physgrad_param_grad physgrad_simple_param physgrad_param_learning physgrad_improved_learning physgrad_time_grad_test physgrad_simple_dt_test physgrad_debug_dt_forces physgrad_direct_dt_test physgrad_trajectory physgrad_simple_traj physgrad_grad_check physgrad_debug_tape

.PHONY: all console inverse validation simple_test stability param_grad simple_param param_learning improved_learning time_grad_test simple_dt_test debug_dt_forces direct_dt_test trajectory simple_traj grad_check debug_tape tests clean