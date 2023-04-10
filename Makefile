CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc

refgpu:
	$(CUDA_COMPILER) -o refgpu refgpu.cu -lcublas -lcudnn 
	./refgpu

refcpu: 
	g++ -o refcpu refcpu.cpp -std=c++11
	./refcpu

archgpu:
	$(CUDA_COMPILER) -o arch arch.cu -lcublas -lcudnn 
	./arch 16

archcpu:
	g++ -o archcpu arch.cpp -std=c++11
	./archcpu	

clean:
	rm -f refgpu refcpu archgpu archcpu

