CXX = g++
NVCC = nvcc
CUDA_PATH = /usr/local/cuda-12.1

CXX_FLAGS = -std=c++17 -O3 -Wall
NVCC_FLAGS = -std=c++17 -arch=sm_75
INCLUDES = -I$(CUDA_PATH)/include -I../src
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lm

all: test_cuda test_physics

test_cuda: test_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

test_physics: test_physics.cpp ../src/simulation.o
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o test_physics.o
	$(NVCC) $(NVCC_FLAGS) -o $@ test_physics.o ../src/simulation.o $(LIBS)

../src/simulation.o:
	cd .. && make src/simulation.o

clean:
	rm -f test_cuda test_physics *.o

.PHONY: all clean