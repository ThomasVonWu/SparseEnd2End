// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <tuple>
#include <vector>

#include "../cuda_wrapper.h"

namespace sparse_end2end {
namespace common {

struct TestType {
  double a;
  float b;
  char c;
  std::int32_t d;
  bool e;
  std::int64_t f;
};

std::ostream& operator<<(std::ostream& stream, const TestType& o) {
  stream << "(" << o.a << ", " << o.b << ", " << o.c << ", " << o.d << ", " << o.e << ", " << o.f << ")";
  return stream;
}

auto integral_equal = [](auto l, auto r) { return l == r; };
auto floating_equal = [](auto l, auto r) { return std::fabs(l - r) < 0.0000001; };
auto test_type_equal = [](const TestType& l, const TestType& r) {
  return floating_equal(l.a, r.a) && floating_equal(l.b, r.b) && l.c == r.c && l.d == r.d && l.e == r.e && l.f == r.f;
};

auto vector_equal = [](const auto& l, const auto& r, auto ele_equal) {
  EXPECT_EQ(l.size(), r.size());
  auto index = 0U;
  for (auto v : l) {
    EXPECT_TRUE(ele_equal(v, r[index])) << "actual: " << v << ", target: " << r[index] << ", at index: " << index;
    index++;
  }
};

struct to_base {
  std::int32_t seed = 0;
};

template <typename T>
struct to : to_base {
  T value() { return static_cast<T>(seed); }
};

template <>
struct to<bool> : to_base {
  bool value() { return seed % 2 == 0 ? true : false; }
};

template <>
struct to<float> : to_base {
  float value() { return static_cast<float>(seed) / 1.0001F; }
};

template <>
struct to<double> : to_base {
  double value() { return static_cast<double>(seed) / 1.0001F; }
};

template <typename T>
std::vector<T> CreateRandom(size_t count, std::int32_t seed = 0) {
  std::vector<T> datas;
  datas.reserve(count);
  if (seed == 0) {
    srandom((unsigned)time(NULL));
  } else {
    srandom((unsigned)seed);
  }
  for (auto i = 0U; i < count; ++i) {
    datas.push_back(to<T>{rand()}.value());
  }

  return datas;
}

template <>
std::vector<TestType> CreateRandom<TestType>(size_t count, std::int32_t seed) {
  std::vector<TestType> datas;
  datas.reserve(count);
  if (seed == 0) {
    srandom((unsigned)time(NULL));
  } else {
    srandom((unsigned)seed);
  }

  for (auto i = 0U; i < count; ++i) {
    datas.push_back(TestType{to<double>{rand()}.value(), to<float>{rand()}.value(), to<char>{rand()}.value(),
                             to<std::int32_t>{rand()}.value(), to<bool>{rand()}.value(),
                             to<std::int64_t>{rand()}.value()});
  }

  return datas;
}

template <typename T, typename Enable = void>
struct equal_traits {
  static constexpr auto compare = integral_equal;
};

template <typename T>
struct equal_traits<T, std::enable_if_t<std::is_floating_point<T>::value, void>> {
  static constexpr auto compare = floating_equal;
};

template <typename T>
struct equal_traits<T, typename std::enable_if_t<std::is_same<T, TestType>::value, void>> {
  static constexpr auto compare = test_type_equal;
};

template <typename T>
void ContructWithSizeTestImpl(const std::uint64_t size) {
  CUDAwrapper<T> data_dev(size);

  if (size == 0) {
    ASSERT_TRUE(data_dev.GetCudaPtr() == nullptr);
  } else {
    ASSERT_TRUE(data_dev.GetCudaPtr() != nullptr);
  }
  ASSERT_TRUE(data_dev.GetSize() == size);
  auto data_host = data_dev.CudaMemcpyCudaToCpu();
  ASSERT_TRUE(data_host.size() == size);
}

template <typename T>
void ConstructWithVectorTestImpl(const std::uint64_t size) {
  std::cout << "ConstructWithVectorTestImpl<" << typeid(T).name() << ">, size: " << size << ", sizeof<"
            << typeid(T).name() << ">: " << sizeof(T) << std::endl;
  auto initials = CreateRandom<T>(size);
  CUDAwrapper<T> data_dev(initials);
  ASSERT_EQ(data_dev.GetSize(), initials.size());

  auto data_host = data_dev.CudaMemcpyCudaToCpu();
  ASSERT_EQ(data_host.size(), initials.size());
  vector_equal(data_host, initials, equal_traits<T>::compare);
}

template <typename T>
void MemResetTestImpl() {
  std::cout << "MemResetTest<" << typeid(T).name() << "> " << std::endl;
  auto nums_count(1000U);
  auto data_host = CreateRandom<T>(nums_count, 0);
  CUDAwrapper<T> data_dev(data_host);

  std::int32_t seed(0);
  if constexpr (std::is_same<T, TestType>::value) {
    seed = data_host[nums_count / 2U].d;
  } else {
    seed = static_cast<std::int32_t>(data_host[nums_count / 2U]);
  }

  auto update_values = CreateRandom<T>(1, seed);
  const auto& reset_value = update_values[0];
  data_dev.Reset(reset_value);

  auto new_host_data = data_dev.CudaMemcpyCudaToCpu();
  for (auto v : new_host_data) {
    EXPECT_TRUE(equal_traits<T>::compare(v, reset_value)) << "actual: " << v << ", target: " << reset_value;
  }
}

template <typename T>
void MemUpdateTestImpl(const std::vector<T>& initials, const std::vector<T>& updates) {
  std::cout << "MemUpdateTest<" << typeid(T).name() << ">, initial size: " << initials.size()
            << ", update size: " << updates.size() << std::endl;
  CUDAwrapper<T> data_dev(initials);
  ASSERT_EQ(data_dev.GetSize(), initials.size());

  data_dev.CudaMemUpdate(updates);
  ASSERT_EQ(data_dev.GetSize(), updates.size());
  auto data_host = data_dev.CudaMemcpyCudaToCpu();
  vector_equal(data_host, updates, equal_traits<T>::compare);
}

template <typename T>
void RandomMemUpdateTest(const std::vector<std::tuple<size_t, size_t>>& size_pairs) {
  for (auto [initial_size, update_size] : size_pairs) {
    auto initials = CreateRandom<T>(initial_size);
    std::int32_t seed(0);
    if constexpr (std::is_same<T, TestType>::value) {
      seed = initials[initial_size / 2U].d;
    } else {
      seed = static_cast<std::int32_t>(initials[initial_size / 2U]);
    }
    auto updates = CreateRandom<T>(update_size, seed);
    MemUpdateTestImpl(initials, updates);
  }
}

template <typename T>
void MemUpdateTwiceTestImpl(const std::vector<T>& initial, const std::vector<T>& second, const std::vector<T>& third) {
  std::cout << "MemUpdateTwiceTest<" << typeid(T).name() << ">, initial size: " << initial.size()
            << ", second size: " << second.size() << ", third size: " << third.size() << std::endl;

  CUDAwrapper<T> data_dev(initial);
  ASSERT_EQ(data_dev.GetSize(), initial.size());

  data_dev.CudaMemUpdate(second);
  ASSERT_EQ(data_dev.GetSize(), second.size());
  auto data_host_2 = data_dev.CudaMemcpyCudaToCpu();
  vector_equal(data_host_2, second, equal_traits<T>::compare);

  data_dev.CudaMemUpdate(third);
  ASSERT_EQ(data_dev.GetSize(), third.size());
  auto data_host_3 = data_dev.CudaMemcpyCudaToCpu();
  vector_equal(data_host_3, third, equal_traits<T>::compare);
}

template <typename T>
void RandomMemUpdateTwiceTest(size_t initial_size, size_t second_size, size_t third_size) {
  auto initials = CreateRandom<T>(initial_size);
  std::int32_t seed_2(0), seed_3(0);
  if constexpr (std::is_same<T, TestType>::value) {
    seed_2 = initials[initial_size / 2U].d;
  } else {
    seed_2 = static_cast<std::int32_t>(initials[initial_size / 2U]);
  }
  seed_3 = seed_2 + 10;

  auto second = CreateRandom<T>(second_size, seed_2);
  auto third = CreateRandom<T>(third_size, seed_3);

  MemUpdateTwiceTestImpl(initials, second, third);
}

/// @brief Variadic Template for CUDAWrapperTester
template <typename... T>
struct CUDAWrapperTester {
  static void ConstructTest(std::uint64_t size);
  static void ConstructWithVectorTest(std::uint64_t size);
  static void ResetTest();
  static void MemUpdateTest();
  static void MemUpdateTwiceTest(size_t first_size, size_t second_size, size_t third_size);
};

/// @brief Partial Specialization Template for CUDAWrapperTester
template <typename T>
struct CUDAWrapperTester<T> {
  static void ConstructTest(std::uint64_t size) { ContructWithSizeTestImpl<T>(size); }

  static void ConstructWithVectorTest(std::uint64_t size) { ConstructWithVectorTestImpl<T>(size); }

  static void ResetTest() { MemResetTestImpl<T>(); }

  static void MemUpdateTest() { RandomMemUpdateTest<T>({{100, 200}, {300, 180}}); }

  static void MemUpdateTwiceTest(size_t first_size, size_t second_size, size_t third_size) {
    RandomMemUpdateTwiceTest<T>(first_size, second_size, third_size);
  }
};

/// @brief Variadic Template for CUDAWrapperTester : Recursion
template <typename T0, typename... T1toN>
struct CUDAWrapperTester<T0, T1toN...> {
  using Other = CUDAWrapperTester<T1toN...>;
  static void ConstructTest(std::uint64_t size) {
    ContructWithSizeTestImpl<T0>(size);
    Other::ConstructTest(size);
  }
  static void ConstructWithVectorTest(std::uint64_t size) {
    ConstructWithVectorTestImpl<T0>(size);
    Other::ConstructWithVectorTest(size);
  }
  static void ResetTest() {
    MemResetTestImpl<T0>();
    Other::ResetTest();
  }

  static void MemUpdateTest() {
    RandomMemUpdateTest<T0>({{100, 200}, {300, 180}});
    Other::MemUpdateTest();
  }

  static void MemUpdateTwiceTest(size_t first_size, size_t second_size, size_t third_size) {
    RandomMemUpdateTwiceTest<T0>(first_size, second_size, third_size);
    Other::MemUpdateTwiceTest(first_size, second_size, third_size);
  }
};

TEST(CUDAwrapper_unit_test, ConstructWithZeroSize) {
  CUDAWrapperTester<bool, std::int16_t, char, unsigned char, std::int32_t, std::uint32_t, std::uint16_t, std::uint8_t,
                    std::int64_t, float, double, long double, TestType>()
      .ConstructTest(0);
}

TEST(CUDAwrapper_unit_test, ConstructWithDesignateSizeTest) {
  CUDAWrapperTester<bool, std::int16_t, char, unsigned char, std::int32_t, uint32_t, uint16_t, uint8_t, std::int64_t,
                    float, double, long double, TestType>()
      .ConstructTest(1000);
}

TEST(CUDAwrapper_unit_test, ConstructWithZeroSizeVectorTest) {
  CUDAWrapperTester<bool, std::int16_t, char, unsigned char, std::int32_t, uint32_t, uint16_t, uint8_t, std::int64_t,
                    float, double, long double, TestType>()
      .ConstructWithVectorTest(0);
}

TEST(CUDAwrapper_unit_test, ConstructWithDesignateSizeVectorTest) {
  CUDAWrapperTester<bool, std::int16_t, char, unsigned char, std::int32_t, uint32_t, uint16_t, uint8_t, std::int64_t,
                    float, double, long double, TestType>()
      .ConstructWithVectorTest(300);
}

TEST(CUDAwrapper_unit_test, ResetTest) {
  CUDAWrapperTester<bool, std::int16_t, char, unsigned char, std::int32_t, uint32_t, uint16_t, uint8_t, std::int64_t,
                    float, double, long double, TestType>()
      .ResetTest();
}

TEST(CUDAwrapper_unit_test, CudaMyMemUpdateTest) {
  CUDAWrapperTester<bool, std::int16_t, char, unsigned char, std::int32_t, uint32_t, uint16_t, uint8_t, std::int64_t,
                    float, double, long double, TestType>()
      .MemUpdateTest();
}

TEST(CUDAwrapper_unit_test, CudaMyMemUpdateTwiceTest) {
  std::vector<std::tuple<size_t, size_t, size_t>> test_sizes = {
      std::make_tuple(100U, 200U, 300U), std::make_tuple(100U, 300U, 200U), std::make_tuple(200U, 300U, 100U),
      std::make_tuple(200U, 100U, 300U), std::make_tuple(300U, 200U, 100U), std::make_tuple(300U, 100U, 200U)};

  for (const auto& [first, second, third] : test_sizes) {
    CUDAWrapperTester<bool, std::int16_t, char, unsigned char, std::int32_t, uint32_t, uint16_t, uint8_t, std::int64_t,
                      float, double, long double, TestType>()
        .MemUpdateTwiceTest(first, second, third);
  }
}

class NoTrivialType {
 public:
  NoTrivialType() { a = 0; }
  std::int32_t a;
};

class NoStandardLayoutType {
 public:
  std::int32_t a;

 private:
  std::int32_t b;
};

TEST(CUDAwrapper_unit_test, NoTrivialTypeUnSupportedTest) {
  CUDAwrapper<NoTrivialType> data_dev;
  ASSERT_TRUE(data_dev.UnSupported());
}

TEST(CUDAwrapper_unit_test, NoStandardLayoutTypeUnSupportedTest) {
  CUDAwrapper<NoStandardLayoutType> data_dev;
  ASSERT_TRUE(data_dev.UnSupported());
}

}  // namespace common
}  // namespace sparse_end2end