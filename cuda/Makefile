CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc
SRC_DIR = src/
all: clean archgpu

refgpu:
	$(CUDA_COMPILER) -o build/refgpu $(SRC_DIR)refgpu.cu -lcublas -lcudnn
	./build/refgpu

refcpu: 
	g++ -o build/refcpu $(SRC_DIR)refcpu.cpp -std=c++11
	./build/refcpu

archgpu:
	$(CUDA_COMPILER) -o build/archgpu $(SRC_DIR)archgpu.cu -lcublas -lcudnn
	# ./build/archgpu NUM_CHANNELS INPUT_WIDTH INPUT_HEIGH FILTER_DIM NUM_RUNS CASE_SELECT
	# If CASE_SELECT == 1, FILTER_DIM is ignored ie integrated
	# ./build/archgpu 2048 13 13 5 50 0
	# ./build/archgpu 512 13 13 5 100 0
	# ./build/archgpu 512 13 13 9 100 0
	# ./build/archgpu 512 13 13 13 100 0
	./build/archgpu 512 13 13 13 1 1

inline_maxpool_integrated:
	$(CUDA_COMPILER) -o build/maxpool_integrated $(SRC_DIR)maxpool_integrated.cu -lcublas -lcudnn

archcpu:
	g++ -o build/archcpu $(SRC_DIR)arch.cpp -std=c++11
	./build/archcpu	

clean:
	rm -f ./build/*
	rm -f *.out

