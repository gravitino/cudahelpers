#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <type_traits>
#include <iostream>
#include <cstdint>

// helper for gcc version check
#define GCC_VERSION (__GNUC__ * 10000                                          \
    + __GNUC_MINOR__ * 100                                                     \
    + __GNUC_PATCHLEVEL__)

// debug prinf
#ifdef DEBUG
    #define STRINGIZE_DETAIL(x) #x
    #define STRINGIZE(x) STRINGIZE_DETAIL(x)
    #define debug_printf(fmt, ...)                                             \
        printf("[DEBUG] file " STRINGIZE(__FILE__)                             \
        ", line " STRINGIZE(__LINE__) ": " STRINGIZE(fmt) "\n",                \
        __VA_ARGS__);
#else
    #define debug_printf(fmt, ...)
#endif

// common CUDA constants
#define WARPSIZE (32)
#define MAXBLOCKSIZE (1024)
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
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
        a##label = std::chrono::system_clock::now();
#else
    #define TIMERSTART(label)                                                  \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);
#endif

#ifndef __CUDACC__
    #define TIMERSTOP(label)                                                   \
        b##label = std::chrono::system_clock::now();                           \
        std::chrono::duration<double> delta##label = b##label-a##label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta##label.count()  << "s" << std::endl;
#else
    #define TIMERSTOP(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms (" << #label << ")" \
                      << std::endl;
#endif

#ifdef __CUDACC__
    /* FIXME
    #define BANDWIDTHSTART(label)                                              \
        cudaSetDevice(0);                                                      \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);

    #define BANDWIDTHSTOP(label, bytes)                                        \
        cudaSetDevice(0);                                                      \
        cudaEventRecord(stop##label, 0);                                       \
        cudaEventSynchronize(stop##label);                                     \
        cudaEventElapsedTime(&time##label, start##label, stop##label);         \
        double bandwidth##label = (bytes)*1000UL/time##label/(1UL<<30);        \
        std::cout << "TIMING: " << time##label << " ms "                       \
                << "-> " << bandwidth##label << " GB/s bandwidth ("            \
                << #label << ")" << std::endl;
    */
    
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

    // negation on value and bit level
    constexpr no_init_t& operator - () noexcept { v_ = -v_; return *this; }
    constexpr no_init_t& operator ~ () noexcept { v_ = ~v_; return *this; }

    // increment/decrement operators
    constexpr no_init_t& operator ++ ()    noexcept { v_++; return *this; }
    constexpr no_init_t& operator ++ (int) noexcept { v_++; return *this; }
    constexpr no_init_t& operator -- ()    noexcept { v_--; return *this; }
    constexpr no_init_t& operator -- (int) noexcept { v_--; return *this; }

    // assignment operators
    constexpr no_init_t& operator  += (T v) noexcept { v_  += v; return *this; }
    constexpr no_init_t& operator  -= (T v) noexcept { v_  -= v; return *this; }
    constexpr no_init_t& operator  *= (T v) noexcept { v_  *= v; return *this; }
    constexpr no_init_t& operator  /= (T v) noexcept { v_  /= v; return *this; }

    // bit-wise operators
    constexpr no_init_t& operator  &= (T v) noexcept { v_  &= v; return *this; }
    constexpr no_init_t& operator  |= (T v) noexcept { v_  |= v; return *this; }
    constexpr no_init_t& operator  ^= (T v) noexcept { v_  ^= v; return *this; }
    constexpr no_init_t& operator >>= (T v) noexcept { v_ >>= v; return *this; }
    constexpr no_init_t& operator <<= (T v) noexcept { v_ <<= v; return *this; }

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

    /* experimental feature (use with compile option --expt-extended-lambda)
    template<class T>
    GLOBALQUALIFIER void generic_kernel(T f)
    {
        f();
    }
    */

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

#endif

#endif /*CUDA_HELPERS_CUH*/
