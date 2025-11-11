CC	= g++
CFLAGS	= -std=c++14 -O3
NVCC	= nvcc
NVCCFLAGS = -std=c++14 -O3

all: ./build/main ./build/omp_fft ./build/cuda_fft

./build/main: main.cpp
	$(CC) $(CFLAGS) -o $@ $<

./build/omp_fft: omp_fft.cpp
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

./build/cuda_fft: cuda_fft.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: clean

clean: 
	rm -f ./build/*

run: ./build/main
	./build/main

run_omp: ./build/omp_fft
	./build/omp_fft

run_cuda: ./build/cuda_fft
	./build/cuda_fft
