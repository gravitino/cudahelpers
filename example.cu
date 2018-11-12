#include <vector>
#include "cuda_helpers.cuh"

#define N ((1L)<<(16))


GLOBALQUALIFIER
void reverse_kernel(int * array, size_t n) {

    size_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    
    if (thid < n/2) {
        const int lower = array[thid];
        const int upper = array[N-thid-1];
        array[thid] = upper;
        array[N-thid-1] = lower;
    }
}

int main () {
    init_cuda_context();                                                  CUERR

    TIMERSTART(allover)

    std::vector<int> host(N);
    for (size_t i = 0; i < N; i++)
        host[i] = i;
    
    int * device = NULL;
    cudaMalloc(&device, sizeof(int)*N);                                   CUERR
    cudaMemcpy(device, host.data(), sizeof(int)*N, H2D);                  CUERR
    
    TIMERSTART(kernel)
    reverse_kernel<<<SDIV(N, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(device, N);   CUERR
    TIMERSTOP(kernel)

    cudaMemcpy(host.data(), device, sizeof(int)*N, D2H);                  CUERR
    
    TIMERSTOP(allover)
    
    std::cout << "causing memory error by allocating 2^60 bytes" << std::endl;
    cudaMalloc(&device, (1L<<60));                                        CUERR
    cudaDeviceSynchronize();
}
