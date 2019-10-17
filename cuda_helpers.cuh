#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <type_traits>
#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <assert.h>

// helper for gcc version check
#define GCC_VERSION (__GNUC__ * 10000                                          \
    + __GNUC_MINOR__ * 100                                                     \
    + __GNUC_PATCHLEVEL__)

// debug prinf
#ifdef _DEBUG
    #define STRINGIZE_DETAIL(x) #x
    #define STRINGIZE(x) STRINGIZE_DETAIL(x)
    #define debug_printf(fmt, ...)                                             \
        printf("[DEBUG] file " STRINGIZE(__FILE__)                             \
        ", line " STRINGIZE(__LINE__) ": " STRINGIZE(fmt) "\n",                \
        ##__VA_ARGS__);
#else
    #define debug_printf(fmt, ...)
#endif

// common CUDA constants
#define WARPSIZE (32)
#define MAXBLOCKSIZE (1024)
#define MAXSMEMBYTES (49152)
#define MAXCONSTMEMBYTES (65536)
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define H2H (cudaMemcpyHostToHost)
#define D2D (cudaMemcpyDeviceToDevice)

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

// convenient timers
#ifndef __CUDACC__
    #include <chrono>
#endif

#ifndef __CUDACC__
    #define TIMERSTART(label)                                                  \
        std::chrono::time_point<std::chrono::system_clock>                     \
            timerstart##label,                                                 \
            timerstop##label;                                                  \
        timerstart##label = std::chrono::system_clock::now();
#else
    #define TIMERSTART(label)                                                  \
        cudaEvent_t timerstart##label, timerstop##label;                       \
        float timerdelta##label;                                               \
        cudaEventCreate(&timerstart##label);                                   \
        cudaEventCreate(&timerstop##label);                                    \
        cudaEventRecord(timerstart##label, 0);
#endif

#ifndef __CUDACC__
    #define TIMERSTOP(label)                                                   \
        stop##label = std::chrono::system_clock::now();                        \
        std::chrono::duration<double>                                          \
            timerdelta##label = timerstop##label-timerstart##label;            \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << timerdelta##label.count()  << "s" << std::endl;
#else
    #define TIMERSTOP(label)                                                   \
            cudaEventRecord(timerstop##label, 0);                              \
            cudaEventSynchronize(timerstop##label);                            \
            cudaEventElapsedTime(                                              \
                &timerdelta##label,                                            \
                timerstart##label,                                             \
                timerstop##label);                                             \
            std::cout <<                                                       \
                "TIMING: " <<                                                  \
                timerdelta##label << " ms (" <<                                \
                #label <<                                                      \
                ")" << std::endl;
#endif

#ifdef __CUDACC__
    #define THROUGHPUTSTART(label)                                             \
        cudaEvent_t throughputstart##label, throughputstop##label;             \
        float throughputdelta##label;                                          \
        cudaEventCreate(&throughputstart##label);                              \
        cudaEventCreate(&throughputstop##label);                               \
        cudaEventRecord(throughputstart##label, 0);

    #define THROUGHPUTSTOP(label, bytes, num)                                  \
        cudaEventRecord(throughputstop##label, 0);                             \
        cudaEventSynchronize(throughputstop##label);                           \
        cudaEventElapsedTime(                                                  \
            &throughputdelta##label,                                           \
            throughputstart##label,                                            \
            throughputstop##label);                                            \
        double throughput##label =                                             \
            (((bytes)*(num))/1073741824.0)/((throughputdelta##label)/1000.0);  \
        double ops##label =                                                    \
            (num)/((throughputdelta##label)/1000.0);                           \
        std::cout <<                                                           \
            "THROUGHPUT: " <<                                                  \
            throughputdelta##label << " ms @ " <<                              \
            ((bytes)*(num))/1073741824.0 << " GB " <<                          \
            "-> " << ops##label << " elements/s or " <<                        \
            throughput##label << " GB/s (" <<                                  \
            #label <<                                                          \
            ")" << std::endl;

    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }
#endif

template<class T>
class no_init_t 
{
public:

    static_assert(std::is_fundamental<T>::value &&
                  std::is_arithmetic<T>::value, 
                  "wrapped type must be a fundamental, numeric type");

    //do nothing
    constexpr no_init_t() noexcept {}

    //convertible from a T
    constexpr no_init_t(T value) noexcept: v_(value) {}

    //act as a T in all conversion contexts
    constexpr operator T () const noexcept { return v_; }

private:
    T v_;
};

// cross platform classifiers
#ifdef __CUDACC__
    #define HOSTDEVICEQUALIFIER  __host__ __device__
#else
    #define HOSTDEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define INLINEQUALIFIER  __forceinline__
#else
    #define INLINEQUALIFIER inline
#endif

#ifdef __CUDACC__
    #define GLOBALQUALIFIER  __global__
#else
    #define GLOBALQUALIFIER
#endif

#ifdef __CUDACC__
    #define DEVICEQUALIFIER  __device__
#else
    #define DEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define HOSTQUALIFIER  __host__
#else
    #define HOSTQUALIFIER
#endif

// redefinition of CUDA atomics for common cstdint types
#ifdef __CUDACC__
    // CAS
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicCAS(
        std::uint64_t* address, 
        std::uint64_t compare, 
        std::uint64_t val)
    {
        return atomicCAS(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(compare),
            static_cast<unsigned long long int>(val));
    }

    // Add
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicAdd(std::uint64_t* address, std::uint64_t val)
    {
        return atomicAdd(
            reinterpret_cast<unsigned long long int*>(address), 
            static_cast<unsigned long long int>(val));
    }

    // Exch
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicExch(std::uint64_t* address, std::uint64_t val)
    {
        return atomicExch(
            reinterpret_cast<unsigned long long int*>(address), 
            static_cast<unsigned long long int>(val));
    }

    // Min
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicMin(std::uint64_t* address, std::uint64_t val)
    {
        return atomicMin(
            reinterpret_cast<unsigned long long int*>(address), 
            static_cast<unsigned long long int>(val));
    }

    // Max
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicMax(std::uint64_t* address, std::uint64_t val)
    {
        return atomicMax(
            reinterpret_cast<unsigned long long int*>(address), 
            static_cast<unsigned long long int>(val));
    }

    // AND
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicAnd(std::uint64_t* address, std::uint64_t val)
    {
        return atomicAnd(
            reinterpret_cast<unsigned long long int*>(address), 
            static_cast<unsigned long long int>(val));
    }

    // OR
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicOr(std::uint64_t* address, std::uint64_t val)
    {
        return atomicOr(
            reinterpret_cast<unsigned long long int*>(address), 
            static_cast<unsigned long long int>(val));
    }

    // XOR
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicXor(std::uint64_t* address, uint64_t val)
    {
        return atomicXor(
            reinterpret_cast<unsigned long long int*>(address), 
            static_cast<unsigned long long int>(val));
    }

    #ifdef __CUDACC_EXTENDED_LAMBDA__
    template<class T>
    GLOBALQUALIFIER void lambda_kernel(T f)
    {
        f();
    }
    #endif

    DEVICEQUALIFIER INLINEQUALIFIER 
    unsigned int lane_id() 
    {
        unsigned int lane;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
        return lane;
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    int ffs(std::uint32_t x)
    {
        return __ffs(x);
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    int ffs(std::uint64_t x)
    {
        return __ffsll(x);
    }

    HOSTQUALIFIER INLINEQUALIFIER
    void init_cuda_context()
    {
        cudaFree(0);
    }

    HOSTQUALIFIER INLINEQUALIFIER
    std::uint64_t available_gpu_memory(float security_factor = 1.0)
    {
        assert(security_factor >= 1.0 && "invalid security factor");
    
        std::uint64_t free;
        std::uint64_t total;
    
        cudaMemGetInfo(&free, &total);
    
        return free / security_factor;
    }
    
    HOSTQUALIFIER INLINEQUALIFIER
    std::vector<std::uint64_t> available_gpu_memory(
        std::vector<std::uint64_t> device_ids, 
        float security_factor = 1.0)
    {
        std::vector<std::uint64_t> available;
    
        for(auto id : device_ids)
        {
            cudaSetDevice(id);
            available.push_back(available_gpu_memory(security_factor));
        }
    
        return available;
    }
    
    HOSTQUALIFIER INLINEQUALIFIER
    std::uint64_t aggregated_available_gpu_memory(
        std::vector<std::uint64_t> device_ids, 
        float security_factor = 1.0,
        bool uniform = false)
    {
        std::sort(device_ids.begin(), device_ids.end());
        device_ids.erase(
            std::unique(device_ids.begin(), device_ids.end()), device_ids.end());
    
        std::vector<std::uint64_t> available = 
            available_gpu_memory(device_ids, security_factor);
    
        if(uniform)
        {
            std::uint64_t min_bytes = 
                *std::min_element(available.begin(), available.end());
    
            return min_bytes * device_ids.size();
        }
        else
        {
            std::uint64_t total = 0;
    
            for(auto bytes : available)
            {
                total += bytes;
            }
    
            return total;
        }
    }

    #if CUDART_VERSION >= 9000
        #include <cooperative_groups.h>

        template<typename index_t>
        DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggInc(index_t * ctr)
        {
            using namespace cooperative_groups;
            coalesced_group g = coalesced_threads();
            index_t prev;
            if (g.thread_rank() == 0) {
                prev = atomicAdd(ctr, g.size());
            }
            prev = g.thread_rank() + g.shfl(prev, 0);
            return prev;
        }
    #else
        template<typename index_t>
        DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggInc(index_t * ctr)
        {
            int lane = lane_id();
            //check if thread is active
            int mask = __ballot(1);
            //determine first active lane for atomic add
            int leader = __ffs(mask) - 1;
            index_t res;
            if (lane == leader) res = atomicAdd(ctr, __popc(mask));
            //broadcast to warp
            res = __shfl(res, leader);
            //compute index for each thread
            return res + __popc(mask & ((1 << lane) -1));
        }
    #endif

    DEVICEQUALIFIER INLINEQUALIFIER
    void die() { assert(0); } // mharris style
#endif

HOSTDEVICEQUALIFIER INLINEQUALIFIER
float B2KB(std::size_t bytes) noexcept { return float(bytes)/1024.0; }

HOSTDEVICEQUALIFIER INLINEQUALIFIER
float B2MB(std::size_t bytes) noexcept { return float(bytes)/1048576.0; }

HOSTDEVICEQUALIFIER INLINEQUALIFIER
float B2GB(std::size_t bytes) noexcept { return float(bytes)/1073741824.0; }

HOSTDEVICEQUALIFIER INLINEQUALIFIER
std::size_t KB2B(float kb) noexcept { return std::size_t(kb*1024); }

HOSTDEVICEQUALIFIER INLINEQUALIFIER
std::size_t MB2B(float mb) noexcept { return std::size_t(mb*1048576); }

HOSTDEVICEQUALIFIER INLINEQUALIFIER
std::size_t GB2B(float gb) noexcept { return std::size_t(gb*1073741824); }

#endif /*CUDA_HELPERS_CUH*/
