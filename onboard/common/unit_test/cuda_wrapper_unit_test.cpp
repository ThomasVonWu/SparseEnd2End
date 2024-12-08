// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <tuple>
#include <vector>

#include "../cuda_wrapper.cu.h"

/* ATTENTION : Not Support bool Type:
 *
 * Error like below:
 * error: passing ‘const std::vector<bool, std::allocator<bool> >’ as ‘this’ argument discards qualifiers [-fpermissive]
 * 185 | checkCudaErrors(cudaMemcpy(cuda_ptr, (const void*)vec_data.data(), size, cudaMemcpyHostToDevice));
 *
 */

namespace sparse_end2end {
namespace common {

/// @brief Different type data compare.
struct TestType {
  double a;
  float b;
  char c;
  std::int32_t d;
  std::uint8_t e;
  std::int64_t f;
};

std::ostream &operator<<(std::ostream &stream, const TestType &o) {
  stream << "(" << o.a << ", " << o.b << ", " << o.c << ", " << o.d << ", " << o.e << ", " << o.f << ")";
  return stream;
}

auto integral_equal = [](auto l, auto r) { return l == r; };

auto floating_equal = [](auto l, auto r) { return std::fabs(l - r) < 1e-7; };

auto test_type_equal = [](const TestType &l, const TestType &r) {
  return floating_equal(l.a, r.a) && floating_equal(l.b, r.b) && l.c == r.c && l.d == r.d && l.e == r.e && l.f == r.f;
};

auto vector_equal = [](const auto &l, const auto &r, auto ele_equal) {
  EXPECT_EQ(l.size(), r.size());
  auto index = 0U;
  for (auto v : l) {
    EXPECT_TRUE(ele_equal(v, r[index])) << "actual: " << v << ", target: " << r[index] << ", at index: " << index;
    index++;
  }
};

template <typename T, typename Enable = void>
struct equalTraits {
  static constexpr auto compare = integral_equal;
};

template <typename T>
struct equalTraits<T, std::enable_if_t<std::is_floating_point<T>::value, void>> {
  static constexpr auto compare = floating_equal;
};

template <typename T>
struct equalTraits<T, typename std::enable_if_t<std::is_same<T, TestType>::value, void>> {
  static constexpr auto compare = test_type_equal;
};

/// @brief Type Convertion.
struct ToBase {
  std::int32_t seed = 0;
};

template <typename T>
struct To : ToBase {
  T value() { return static_cast<T>(seed); }
};

template <>
struct To<bool> : ToBase {
  bool value() { return seed % 2 == 0 ? true : false; }
};

template <>
struct To<float> : ToBase {
  float value() { return static_cast<float>(seed) / 1.0001F; }
};

template <>
struct To<double> : ToBase {
  double value() { return static_cast<double>(seed) / 1.0001F; }
};

/// @brief  Create random data Implementation for cuda memory Test.
template <typename T>
std::vector<T> createRandom(size_t count, std::int32_t seed = 0) {
  std::vector<T> datas;
  datas.reserve(count);
  if (seed == 0) {
    srandom((std::uint32_t)time(NULL));
  } else {
    srandom((std::uint32_t)seed);
  }
  for (auto i = 0U; i < count; ++i) {
    datas.push_back(To<T>{rand()}.value());
  }

  return datas;
}

template <>
std::vector<TestType> createRandom<TestType>(size_t count, std::int32_t seed) {
  std::vector<TestType> datas;
  datas.reserve(count);
  if (seed == 0) {
    srandom((std::uint32_t)time(NULL));
  } else {
    srandom((std::uint32_t)seed);
  }

  for (auto i = 0U; i < count; ++i) {
    datas.push_back(TestType{To<double>{rand()}.value(), To<float>{rand()}.value(), To<char>{rand()}.value(),
                             To<std::int32_t>{rand()}.value(), To<std::uint8_t>{rand()}.value(),
                             To<std::int64_t>{rand()}.value()});
  }

  return datas;
}

/// @brief Construct function test implementation.
template <typename T>
void constructWithSizeImpl(const std::uint64_t &size) {
  CudaWrapper<T> cudaWrapperObj(size);

  if (size == 0) {
    ASSERT_TRUE(cudaWrapperObj.getCudaPtr() == nullptr);
  } else {
    ASSERT_TRUE(cudaWrapperObj.getCudaPtr() != nullptr);
  }
  ASSERT_TRUE(cudaWrapperObj.getSize() == size);
  std::vector<T> vecData = cudaWrapperObj.cudaMemcpyD2HResWrap();
  ASSERT_TRUE(vecData.size() == size);
}

template <typename T>
void constructWithVecImpl(const std::uint64_t &size) {
  std::cout << "constructWithVecImpl<" << typeid(T).name() << ">, size: " << size << ", sizeof<" << typeid(T).name()
            << ">: " << sizeof(T) << std::endl;
  std::vector<T> initials = createRandom<T>(size);
  CudaWrapper<T> cudaWrapperObj(initials);
  ASSERT_EQ(cudaWrapperObj.getSize(), initials.size());

  std::vector<T> vecdata = cudaWrapperObj.cudaMemcpyD2HResWrap();
  ASSERT_EQ(vecdata.size(), initials.size());
  vector_equal(vecdata, initials, equalTraits<T>::compare);
}

/// @brief Cuda memory set test implementation.
template <typename T>
void cudaMemSetImpl() {
  std::cout << "cudaMemSetImpl<" << typeid(T).name() << "> " << std::endl;
  size_t nums_count(1000U);
  std::vector<T> vecdata = createRandom<T>(nums_count, 0);
  CudaWrapper<T> cudaWrapperObj(vecdata);

  std::int32_t seed(0);
  if constexpr (std::is_same<T, TestType>::value) {
    seed = vecdata[nums_count / 2U].d;
  } else {
    seed = static_cast<std::int32_t>(vecdata[nums_count / 2U]);
  }

  std::vector<T> update_values = createRandom<T>(1, seed);
  const auto &reset_value = update_values[0];
  cudaWrapperObj.cudaMemSetWrap(reset_value);

  std::vector<T> new_vecdata = cudaWrapperObj.cudaMemcpyD2HResWrap();
  for (auto v : new_vecdata) {
    EXPECT_TRUE(equalTraits<T>::compare(v, reset_value)) << "actual: " << v << ", target: " << reset_value;
  }
}

/// @brief Cuda memory update H2D test implementation.
template <typename T>
void randomCudaMemUpdateImpl(const std::vector<T> &initials, const std::vector<T> &updates) {
  std::cout << "randomCudaMemUpdateImpl<" << typeid(T).name() << ">, initial size: " << initials.size()
            << ", update size: " << updates.size() << std::endl;
  CudaWrapper<T> cudaWrapperObj(initials);
  ASSERT_EQ(cudaWrapperObj.getSize(), initials.size());

  cudaWrapperObj.cudaMemUpdateWrap(updates);
  ASSERT_EQ(cudaWrapperObj.getSize(), updates.size());
  std::vector<T> vecdata = cudaWrapperObj.cudaMemcpyD2HResWrap();
  vector_equal(vecdata, updates, equalTraits<T>::compare);
}

template <typename T>
void randomCudaMemUpdateTest(const std::vector<std::tuple<size_t, size_t>> &size_pairs) {
  for (auto [initial_size, update_size] : size_pairs) {
    auto initials = createRandom<T>(initial_size);
    std::int32_t seed(0);
    if constexpr (std::is_same<T, TestType>::value) {
      seed = initials[initial_size / 2U].d;
    } else {
      seed = static_cast<std::int32_t>(initials[initial_size / 2U]);
    }
    auto updates = createRandom<T>(update_size, seed);
    randomCudaMemUpdateImpl(initials, updates);
  }
}

/// @brief Cuda memory update twice H2D test implementation.
template <typename T>
void randomCudaMemUpdateTwiceImpl(const std::vector<T> &initial,
                                  const std::vector<T> &second,
                                  const std::vector<T> &third) {
  std::cout << "randomCudaMemUpdateTwiceImpl<" << typeid(T).name() << ">, initial size: " << initial.size()
            << ", second size: " << second.size() << ", third size: " << third.size() << std::endl;

  CudaWrapper<T> cudaWrapperObj(initial);
  ASSERT_EQ(cudaWrapperObj.getSize(), initial.size());

  cudaWrapperObj.cudaMemUpdateWrap(second);
  ASSERT_EQ(cudaWrapperObj.getSize(), second.size());
  auto vecdata_2 = cudaWrapperObj.cudaMemcpyD2HResWrap();
  vector_equal(vecdata_2, second, equalTraits<T>::compare);

  cudaWrapperObj.cudaMemUpdateWrap(third);
  ASSERT_EQ(cudaWrapperObj.getSize(), third.size());
  auto vecdata_3 = cudaWrapperObj.cudaMemcpyD2HResWrap();
  vector_equal(vecdata_3, third, equalTraits<T>::compare);
}

template <typename T>
void randomCudaMemUpdateTwiceTest(size_t initial_size, size_t second_size, size_t third_size) {
  auto initial = createRandom<T>(initial_size);
  std::int32_t seed_2(0), seed_3(0);
  if constexpr (std::is_same<T, TestType>::value) {
    seed_2 = initial[initial_size / 2U].d;
  } else {
    seed_2 = static_cast<std::int32_t>(initial[initial_size / 2U]);
  }
  seed_3 = seed_2 + 10;

  auto second = createRandom<T>(second_size, seed_2);
  auto third = createRandom<T>(third_size, seed_3);

  randomCudaMemUpdateTwiceImpl(initial, second, third);
}

/// @brief Type1: Variadic Template for CudaWrapperTester
template <typename... T>
struct CudaWrapperTester {
  static void constructFuncTest(std::uint64_t size);
  static void constructFuncWithVecTest(std::uint64_t size);
  static void cudaMemResetTest();
  static void cudaMemUpdateTest();
  static void cudaMemUpdateTwiceTest(size_t first_size, size_t second_size, size_t third_size);
};

/// @brief  Type2: Partial Specialization Template for CudaWrapperTester
template <typename T>
struct CudaWrapperTester<T> {
  static void constructFuncTest(std::uint64_t size) { constructWithSizeImpl<T>(size); }

  static void constructFuncWithVecTest(std::uint64_t size) { constructWithVecImpl<T>(size); }

  static void cudaMemResetTest() { cudaMemSetImpl<T>(); }

  static void cudaMemUpdateTest() { randomCudaMemUpdateTest<T>({{100, 200}, {300, 180}}); }

  static void cudaMemUpdateTwiceTest(size_t first_size, size_t second_size, size_t third_size) {
    randomCudaMemUpdateTwiceTest<T>(first_size, second_size, third_size);
  }
};

/// @brief Type3: Variadic Template for CudaWrapperTester : Recursion
template <typename T0, typename... T1toN>
struct CudaWrapperTester<T0, T1toN...> {
  using Other = CudaWrapperTester<T1toN...>;

  static void constructFuncTest(std::uint64_t size) {
    constructWithSizeImpl<T0>(size);
    Other::constructFuncTest(size);
  }

  static void constructFuncWithVecTest(std::uint64_t size) {
    constructWithVecImpl<T0>(size);
    Other::constructFuncWithVecTest(size);
  }
  static void cudaMemResetTest() {
    cudaMemSetImpl<T0>();
    Other::cudaMemResetTest();
  }

  static void cudaMemUpdateTest() {
    randomCudaMemUpdateTest<T0>({{100, 200}, {300, 180}});
    Other::cudaMemUpdateTest();
  }

  static void cudaMemUpdateTwiceTest(size_t first_size, size_t second_size, size_t third_size) {
    randomCudaMemUpdateTwiceTest<T0>(first_size, second_size, third_size);
    Other::cudaMemUpdateTwiceTest(first_size, second_size, third_size);
  }
};

TEST(CUDAwrapperUnitTest, ConstructWithZeroSize) {
  // std::cout << "---------------------------" << std::endl;
  CudaWrapperTester<char, unsigned char, std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t,
                    std::uint16_t, std::uint32_t, std::uint64_t, float, double, long double, TestType>()
      .constructFuncTest(0);
}

TEST(CUDAwrapperUnitTest, ConstructWithDesignateSizeTest) {
  CudaWrapperTester<char, unsigned char, std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t,
                    std::uint16_t, std::uint32_t, std::uint64_t, float, double, long double, TestType>()
      .constructFuncTest(1000);
}

TEST(CUDAwrapperUnitTest, ConstructWithZeroSizeVectorTest) {
  CudaWrapperTester<char, unsigned char, std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t,
                    std::uint16_t, std::uint32_t, std::uint64_t, float, double, long double, TestType>()
      .constructFuncWithVecTest(0);
}

TEST(CUDAwrapperUnitTest, ConstructWithDesignateSizeVectorTest) {
  CudaWrapperTester<char, unsigned char, std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t,
                    std::uint16_t, std::uint32_t, std::uint64_t, float, double, long double, TestType>()
      .constructFuncWithVecTest(300);
}

TEST(CUDAwrapperUnitTest, CudaMemResetTest) {
  CudaWrapperTester<char, unsigned char, std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t,
                    std::uint16_t, std::uint32_t, std::uint64_t, float, double, long double, TestType>()
      .cudaMemResetTest();
}

TEST(CUDAwrapperUnitTest, CudaMemUpdateTest) {
  CudaWrapperTester<char, unsigned char, std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t,
                    std::uint16_t, std::uint32_t, std::uint64_t, float, double, long double, TestType>()
      .cudaMemUpdateTest();
}

TEST(CUDAwrapperUnitTest, CudaMemUpdateTwiceTest) {
  std::vector<std::tuple<size_t, size_t, size_t>> test_sizes = {
      std::make_tuple(100U, 200U, 300U), std::make_tuple(100U, 300U, 200U), std::make_tuple(200U, 300U, 100U),
      std::make_tuple(200U, 100U, 300U), std::make_tuple(300U, 200U, 100U), std::make_tuple(300U, 100U, 200U)};

  for (const auto &[first, second, third] : test_sizes) {
    CudaWrapperTester<char, unsigned char, std::int8_t, std::int16_t, std::int32_t, std::int64_t, std::uint8_t,
                      std::uint16_t, std::uint32_t, std::uint64_t, float, double, long double, TestType>()
        .cudaMemUpdateTwiceTest(first, second, third);
  }
}

}  // namespace common
}  // namespace sparse_end2end