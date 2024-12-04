// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

#ifndef ONBOARD_COMMON_CUDA_WRAPPER_CU_H
#define ONBOARD_COMMON_CUDA_WRAPPER_CU_H

#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>
#include <vector>

namespace sparse_end2end {
namespace common {
/*
#define checkCudaErrors(status)                                                                                      \
do {                                                                                                               \
    auto ret = (status);                                                                                             \
    if (ret != 0) {                                                                                                  \
    std::cout << "Cuda failure: " << cudaGetErrorString(ret) << " at line " << __LINE__ << " in file " << __FILE__ \
                << " error status: " << ret << "\n";                                                                 \
    abort();                                                                                                       \
    }                                                                                                                \
} while (0)
*/

#define checkCudaErrors(val)                                                                     \
  {                                                                                              \
    cudaError_t err = (val);                                                                     \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(err);                                                                                 \
    }                                                                                            \
  }

template <typename T, typename Enable = void>
struct CudaWrapper;

template <class T>
class CudaWrapper<T, typename std::enable_if_t<std::is_trivial<T>::value && std::is_standard_layout<T>::value, void>> {
 public:
  /// @brief Delete default copy constructor.
  CudaWrapper(const CudaWrapper &cudaWrapper) = delete;
  /// @brief Delete default copy assignment operator.
  CudaWrapper &operator=(const CudaWrapper &cudaWrapper) = delete;

  /// @brief Default Constructor.
  /// @param ptr_ Iner CUDA pointer=nullptr.
  CudaWrapper() {
    size_ = 0U;
    capacity_ = 0U;
    ptr_ = nullptr;
  }

  /// @brief Construct a CUDA object with size : malloc memory with size and memset 0.
  CudaWrapper(std::uint64_t size) {
    if (size != 0) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
      checkCudaErrors(cudaMemset(ptr_, 0, getSizeBytes()));
    }
  }

  /// @brief Construct a CUDA object with given cpu pointer.
  CudaWrapper(T *cpu_ptr, std::uint64_t size) {
    if (size != 0) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
      checkCudaErrors(cudaMemcpy(ptr_, (void *)cpu_ptr, getSizeBytes(), cudaMemcpyHostToDevice));
    }
  }

  /// @brief Construct a CUDA object with given vec_data in cpu and size.
  CudaWrapper(const std::vector<T> &vec_data, std::uint64_t size) {
    if (vec_data.size() >= size) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
      cudaMemcpyH2D(vec_data, ptr_, getSizeBytes());
    }
  }

  /// @brief Construct a CUDA object with given vec_data in cpu only.
  CudaWrapper(const std::vector<T> &vec_data) : CudaWrapper(vec_data, vec_data.size()) {}

  /// @brief Moving Construct for improving efficiency in resource management.
  CudaWrapper(CudaWrapper &&cudaWrapper) {
    size_ = cudaWrapper.size_;
    capacity_ = cudaWrapper.capacity_;
    ptr_ = cudaWrapper.ptr_;

    cudaWrapper.size_ = 0U;
    cudaWrapper.capacity_ = 0U;
    cudaWrapper.ptr_ = nullptr;
  }

  /// @brief Moving assignment operator for improving efficiency in resource management.
  CudaWrapper &operator=(CudaWrapper &&cudaWrapper) {
    if (this != &cudaWrapper) {
      size_ = cudaWrapper.size_;
      capacity_ = cudaWrapper.capacity_;
      ptr_ = cudaWrapper.ptr_;

      cudaWrapper.size_ = 0U;
      cudaWrapper.capacity_ = 0U;
      cudaWrapper.ptr_ = nullptr;
    }
    return *this;
  }

  ~CudaWrapper() {
    size_ = 0U;
    capacity_ = 0U;
    if (ptr_ != nullptr) {
      checkCudaErrors(cudaFree(ptr_));
      ptr_ = nullptr;
    }
  }

  /// @brief Copy data to current CUDAobject ptr_ from CudaWrapper's CUDA memory ptr_.
  void cudaMemcpyD2DWrap(const T *src_cuda_ptr, const std::uint64_t size) {
    if (size > capacity_) {
      checkCudaErrors(cudaFree(ptr_));
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
    } else {
      size_ = size;
    }
    cudaMemcpyD2D(src_cuda_ptr);
  }

  /// @brief Copy data to current CUDAobject ptr_ from CPU memory(vector), private ptr_ will be changed.
  void cudaMemUpdateWrap(const std::vector<T> &vec_data) {
    if (vec_data.size() > capacity_) {
      checkCudaErrors(cudaFree(ptr_));
      size_ = vec_data.size();
      capacity_ = vec_data.size();
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
    } else {
      size_ = vec_data.size();
    }

    cudaMemcpyH2D(vec_data, ptr_, getSizeBytes());
  }

  /// @brief Copy data to CPU memory(vector) from current GPU memory(ptr_).
  /// @param
  /// @return std::vector<T>
  std::vector<T> cudaMemcpyD2HResWrap() const {
    if (size_ == 0) {
      return {};
    }

    return cudaMemcpyD2HRes();
  }

  /// @brief Copy data to given CPU memory(vector) from curren GPU memory(ptr_).
  /// @param buf, std::vector<T>
  /// @return
  void cudaMemcpyD2HParamWrap(const std::vector<T> &buf) const {
    if (buf.size() < size_) {
      return;
    }
    cudaMemcpyD2HParam(buf);
  }

  inline void cudaMemSetWrap() { checkCudaErrors(cudaMemset(ptr_, 0, getSizeBytes())); }

  void cudaMemSetWrap(const T &value) {
    std::vector<T> vecdata(size_, value);
    checkCudaErrors(cudaMemcpy(ptr_, (void *)vecdata.data(), getSizeBytes(), cudaMemcpyHostToDevice));
  }

  inline T *getCudaPtr() const { return ptr_; }

  inline std::uint64_t getSize() const { return size_; }

 private:
  inline std::uint64_t getSizeBytes() const { return size_ * sizeof(std::remove_pointer_t<decltype(ptr_)>); }

  inline void cudaMemcpyD2D(const T *src_cuda_ptr) const {
    checkCudaErrors(cudaMemcpy(ptr_, (void *)src_cuda_ptr, getSizeBytes(), cudaMemcpyDeviceToDevice));
  }

  inline void cudaMemcpyH2D(const std::vector<T> &vec_data, T *cuda_ptr, const std::uint64_t &size) {
    checkCudaErrors(cudaMemcpy(cuda_ptr, (void *)vec_data.data(), size, cudaMemcpyHostToDevice));
  }

  inline std::vector<T> cudaMemcpyD2HRes() const {
    std::vector<T> buf(size_);
    checkCudaErrors(cudaMemcpy((void *)buf.data(), ptr_, getSizeBytes(), cudaMemcpyDeviceToHost));

    return buf;
  }

  inline void cudaMemcpyD2HParam(const std::vector<T> &buf) const {
    checkCudaErrors(cudaMemcpy((void *)buf.data(), ptr_, getSizeBytes(), cudaMemcpyDeviceToHost));
  }

 private:
  std::uint64_t size_ = 0U;
  std::uint64_t capacity_ = 0U;
  T *ptr_ = nullptr;
};

}  // namespace common
}  // namespace sparse_end2end

#endif  // ONBOARD_COMMON_CUDA_WRAPPER_CU_H
