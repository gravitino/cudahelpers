#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <iostream>
#include <cstdint>

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
#endif#include <iostream>
5
​
6
// safe division
7
#define SDIV(x,y)(((x)+(y)-1)/(y))
8
​
9
// error makro
10
#define CUERR {                                                              \
11
    cudaError_t err;                                                         \
12
    if ((err = cudaGetLastError()) != cudaSuccess) {                         \
13
       std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "       \
14
                 << __FILE__ << ", line " << __LINE__ << std::endl;          \
15
       exit(1);                                                              \
16
    }                                                                        \
17
}
18
​
19
// convenient timers
20
#define TIMERSTART(label)                                                    \
21
        cudaEvent_t start##label, stop##label;                               \
22
        float time##label;                                                   \
23
        cudaEventCreate(&start##label);                                      \
24
        cudaEventCreate(&stop##label);                                       \
25
        cudaEventRecord(start##label, 0);
26
​
27
#define TIMERSTOP(label)                                                     \
28
        cudaEventRecord(stop##label, 0);                                     \
29
        cudaEventSynchronize(stop##label);                                   \
30
        cudaEventElapsedTime(&time##label, start##label, stop##label);       \
31
        std::cout << time##label << " ms (" << #label << ")" << std::endl;
32



#ifdef __CUDACC__
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }

    // transfer constants
    #define H2D (cudaMemcpyHostToDevice)
    #define D2H (cudaMemcpyDeviceToHost)
    #define H2H (cudaMemcpyHostToHost)
    #define D2D (cudaMemcpyDeviceToDevice)
#endif

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

#include <type_traits>
#include <iostream>


template<class T>
class no_init_t {
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

#endif
