// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_COMMON_CUDA_WRAPPER_H
#define ONBOARD_COMMON_CUDA_WRAPPER_H
#include <iostream>
#include <type_traits>
#include <vector>

#include "cuda_runtime.h"

namespace sparse_end2end {
namespace common {

#define checkCudaErrors(status)                                                                                      \
  do {                                                                                                               \
    auto ret = (status);                                                                                             \
    if (ret != 0) {                                                                                                  \
      std::cout << "Cuda failure: " << cudaGetErrorString(ret) << " at line " << __LINE__ << " in file " << __FILE__ \
                << " error status: " << ret << "\n";                                                                 \
      abort();                                                                                                       \
    }                                                                                                                \
  } while (0)

template <typename T, typename Enable = void>
struct CUDAwrapper;

template <class T>
class CUDAwrapper<T, typename std::enable_if_t<std::is_trivial<T>::value && std::is_standard_layout<T>::value, void>> {
 public:
  /// @brief Delete copy constructor.
  CUDAwrapper(const CUDAwrapper& cudawrapper) = delete;
  /// @brief Delete copy assignment operator.
  CUDAwrapper& operator=(const CUDAwrapper& rh) = delete;

  /// @brief Default Constructor.
  /// @param ptr_ Iner CUDA pointer=nullptr.
  CUDAwrapper() {
    size_ = 0U;
    capacity_ = 0U;
    ptr_ = nullptr;
  }

  /// @brief Construct a CUDA object with size : malloc memory with size and memset 0.
  CUDAwrapper(std::uint64_t size) {
    if (size != 0) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void**)&ptr_, get_sizebytes()));
      checkCudaErrors(cudaMemset(ptr_, 0, get_sizebytes()));
    }
  }

  /// @brief Construct a CUDA object with given cpu pointer.
  CUDAwrapper(T* cpu_ptr, std::uint64_t size) {
    if (size != 0) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void**)&ptr_, get_sizebytes()));
      checkCudaErrors(cudaMemcpy(ptr_, (void*)cpu_ptr, get_sizebytes(), cudaMemcpyHostToDevice));
    }
  }

  /// @brief Construct a CUDA object with given vec_data in cpu and size.
  CUDAwrapper(const std::vector<T>& vec_data, std::uint64_t size) {
    if (vec_data.size() >= size) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void**)&ptr_, get_sizebytes()));
      vecdata_cpu2gpu(vec_data, ptr_, get_sizebytes());
    }
  }

  /// @brief Construct a CUDA object with given vec_data in cpu only.
  CUDAwrapper(const std::vector<T>& vec_data) : CUDAwrapper(vec_data, vec_data.size()) {}

  /// @brief Moving Construct for improving efficiency in resource management.
  CUDAwrapper(CUDAwrapper&& cudawrapper) {
    size_ = cudawrapper.size_;
    capacity_ = cudawrapper.capacity_;
    ptr_ = cudawrapper.ptr_;

    cudawrapper.size_ = 0U;
    cudawrapper.capacity_ = 0U;
    cudawrapper.ptr_ = nullptr;
  }

  /// @brief Moving assignment operator for improving efficiency in resource management.
  CUDAwrapper& operator=(CUDAwrapper&& cudawrapper) {
    if (this != &cudawrapper) {
      size_ = cudawrapper.size_;
      capacity_ = cudawrapper.capacity_;
      ptr_ = cudawrapper.ptr_;

      cudawrapper.size_ = 0U;
      cudawrapper.capacity_ = 0U;
      cudawrapper.ptr_ = nullptr;
    }
    return *this;
  }

  ~CUDAwrapper() {
    size_ = 0U;
    capacity_ = 0U;
    if (ptr_ != nullptr) {
      checkCudaErrors(cudaFree(ptr_));
      ptr_ = nullptr;
    }
  }

  /// @brief Copy data to current CUDAobject ptr_ from another CUDA memory.
  void cudaMemcpy2D_wrapper(const T* cuda_ptr, const std::uint64_t size) {
    if (size > capacity_) {
      checkCudaErrors(cudaFree(ptr_));
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void**)&ptr_, get_sizebytes()));
    } else {
      size_ = size;
    }

    checkCudaErrors(cudaMemcpy(ptr_, (void*)cuda_ptr, get_sizebytes(), cudaMemcpyDeviceToDevice));
  }

  /// @brief Copy data to current CUDAobject ptr_ from CPU memory(vector).
  void cudaMemcpy2D_wrapper(const std::vector<T>& vec_data) { cudaMemcpy2D_wrapper(vec_data, vec_data.size()); }

  /// @brief Copy data to current CUDAobject ptr_ from CPU memory(vector) with given size.
  void cudaMemcpy2D_wrapper(const std::vector<T>& vec_data, std::uint64_t size) {
    if (vec_data.size() < size) {
      return;
    }

    if (size > capacity_) {
      checkCudaErrors(cudaFree(ptr_));
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void**)&ptr_, get_sizebytes()));
    } else {
      size_ = size;
    }

    vecdata_cpu2gpu(vec_data, ptr_, get_sizebytes());
  }

  /// @brief Copy data to CPU memory(vector) from current  GPU memory(ptr_).
  std::vector<T> cudaMemcpy2H_wrapper() const {
    if (size_ == 0) {
      return {};
    }

    return vecdata_gpu2cpu();
  }

  /// @brief Copy data to given CPU memory(vector) from curren GPU memory(ptr_).
  void cudaMemcpy2H_wrapper(std::vector<T>& buf) const {
    if (buf.size() < size_) {
      return;
    }

    cudaMemcpy((void*)buf.data(), ptr_, get_sizebytes(), cudaMemcpyDeviceToHost);
  }

  inline void cudaMemset_wrapper() { cudaMemset(ptr_, 0, get_sizebytes()); }

  void cudaMemset_wrapper(const T& value) {
    std::vector<T> src_data_host(size_, value);
    checkCudaErrors(cudaMemcpy(ptr_, (void*)src_data_host.data(), get_sizebytes(), cudaMemcpyHostToDevice));
  }

  inline T* getCudaPtr() const { return ptr_; }

  inline std::uint64_t getSize() const { return size_; }

 private:
  inline std::uint64_t get_sizebytes() const { return size_ * sizeof(std::remove_ptr_t<decltype(ptr_)>); }

  inline void vecdata_cpu2gpu(const std::vector<T>& vec_data, T* cuda_ptr, std::uint64_t size) {
    checkCudaErrors(cudaMemcpy(cuda_ptr, (void*)vec_data.data(), size, cudaMemcpyHostToDevice));
  }

  inline std::vector<T> vecdata_gpu2cpu() const {
    std::vector<T> buf(size_);
    cudaMemcpy((void*)buf.data(), ptr_, get_sizebytes(), cudaMemcpyDeviceToHost);
    return buf;
  }

 private:
  std::uint64_t size_ = 0U;
  std::uint64_t capacity_ = 0U;
  T* ptr_ = nullptr;
};

}  // namespace common
}  // namespace sparse_end2end

#endif  // ONBOARD_COMMON_CUDA_WRAPPER_H
