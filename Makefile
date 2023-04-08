CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc

refgpu:
	$(CUDA_COMPILER) -o refgpu refgpu.cu -lcublas -lcudnn 
	./refgpu

refcpu: 
	g++ -o refcpu refcpu.cpp -std=c++11
	./refcpu
