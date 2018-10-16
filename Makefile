all: example

example: example.cu cuda_helpers.cuh
	nvcc -arch=sm_35 example.cu -o example -O3

clean:
	rm -rf example
