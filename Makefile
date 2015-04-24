all: example

example: example.cu cuda_helpers.cuh
	nvcc example.cu -o example -O3

clean:
	rm -rf example
