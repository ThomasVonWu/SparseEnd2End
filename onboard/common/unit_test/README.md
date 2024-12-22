
## Unit Test for `cuda_wrapper_unit_test`

```bash
# 1st Step
cd onboard/common/unit_test
cmake -B build -S .
```

You will get log like this:
```bash
-- The CUDA compiler identification is NVIDIA 11.6.55
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Found CUDA: /usr/local/cuda (found suitable version "11.6", minimum required is "11") 
-- The C compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/thomasvowu/PublishRepos/SparseEnd2End/onboard/common/unit_test/build
```

```bash
# 2nd Step
cmake --build build
```

You will get log like this:
```bash
[ 10%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
[ 20%] Linking CXX static library ../../../lib/libgtest.a
[ 20%] Built target gtest
[ 30%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock.dir/src/gmock-all.cc.o
[ 40%] Linking CXX static library ../../../lib/libgmock.a
[ 40%] Built target gmock
[ 50%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
[ 60%] Linking CXX static library ../../../lib/libgmock_main.a
[ 60%] Built target gmock_main
[ 70%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o
[ 80%] Linking CXX static library ../../../lib/libgtest_main.a
[ 80%] Built target gtest_main
[ 90%] Building CXX object CMakeFiles/cuda_wrapper_unit_test.bin.dir/cuda_wrapper_unit_test.cpp.o
[100%] Linking CXX executable bin/cuda_wrapper_unit_test.bin
[100%] Built target cuda_wrapper_unit_test.bin
```

```bash
# 3rd Step
./build/bin/cuda_wrapper_unit_test.bin
```

```
You will get log like this:
```bash
Running main() from /home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/third_party/googletest/googletest/src/gtest_main.cc
[==========] Running 7 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 7 tests from CUDAwrapperUnitTest
[ RUN      ] CUDAwrapperUnitTest.ConstructWithZeroSize
[       OK ] CUDAwrapperUnitTest.ConstructWithZeroSize (0 ms)
[ RUN      ] CUDAwrapperUnitTest.ConstructWithDesignateSizeTest
[       OK ] CUDAwrapperUnitTest.ConstructWithDesignateSizeTest (149 ms)
[ RUN      ] CUDAwrapperUnitTest.ConstructWithZeroSizeVectorTest
constructWithVecImpl<c>, size: 0, sizeof<c>: 1
constructWithVecImpl<h>, size: 0, sizeof<h>: 1
constructWithVecImpl<a>, size: 0, sizeof<a>: 1
constructWithVecImpl<s>, size: 0, sizeof<s>: 2
constructWithVecImpl<i>, size: 0, sizeof<i>: 4
constructWithVecImpl<l>, size: 0, sizeof<l>: 8
constructWithVecImpl<h>, size: 0, sizeof<h>: 1
constructWithVecImpl<t>, size: 0, sizeof<t>: 2
constructWithVecImpl<j>, size: 0, sizeof<j>: 4
constructWithVecImpl<m>, size: 0, sizeof<m>: 8
constructWithVecImpl<f>, size: 0, sizeof<f>: 4
constructWithVecImpl<d>, size: 0, sizeof<d>: 8
constructWithVecImpl<e>, size: 0, sizeof<e>: 16
constructWithVecImpl<N14sparse_end2end6common8TestTypeE>, size: 0, sizeof<N14sparse_end2end6common8TestTypeE>: 32
[       OK ] CUDAwrapperUnitTest.ConstructWithZeroSizeVectorTest (0 ms)
[ RUN      ] CUDAwrapperUnitTest.ConstructWithDesignateSizeVectorTest
constructWithVecImpl<c>, size: 300, sizeof<c>: 1
constructWithVecImpl<h>, size: 300, sizeof<h>: 1
constructWithVecImpl<a>, size: 300, sizeof<a>: 1
constructWithVecImpl<s>, size: 300, sizeof<s>: 2
constructWithVecImpl<i>, size: 300, sizeof<i>: 4
constructWithVecImpl<l>, size: 300, sizeof<l>: 8
constructWithVecImpl<h>, size: 300, sizeof<h>: 1
constructWithVecImpl<t>, size: 300, sizeof<t>: 2
constructWithVecImpl<j>, size: 300, sizeof<j>: 4
constructWithVecImpl<m>, size: 300, sizeof<m>: 8
constructWithVecImpl<f>, size: 300, sizeof<f>: 4
constructWithVecImpl<d>, size: 300, sizeof<d>: 8
constructWithVecImpl<e>, size: 300, sizeof<e>: 16
constructWithVecImpl<N14sparse_end2end6common8TestTypeE>, size: 300, sizeof<N14sparse_end2end6common8TestTypeE>: 32
[       OK ] CUDAwrapperUnitTest.ConstructWithDesignateSizeVectorTest (1 ms)
[ RUN      ] CUDAwrapperUnitTest.CudaMemResetTest
cudaMemSetImpl<c> 
cudaMemSetImpl<h> 
cudaMemSetImpl<a> 
cudaMemSetImpl<s> 
cudaMemSetImpl<i> 
cudaMemSetImpl<l> 
cudaMemSetImpl<h> 
cudaMemSetImpl<t> 
cudaMemSetImpl<j> 
cudaMemSetImpl<m> 
cudaMemSetImpl<f> 
cudaMemSetImpl<d> 
cudaMemSetImpl<e> 
cudaMemSetImpl<N14sparse_end2end6common8TestTypeE> 
[       OK ] CUDAwrapperUnitTest.CudaMemResetTest (2 ms)
[ RUN      ] CUDAwrapperUnitTest.CudaMemUpdateTest
randomCudaMemUpdateImpl<c>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<c>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<h>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<h>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<a>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<a>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<s>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<s>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<i>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<i>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<l>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<l>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<h>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<h>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<t>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<t>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<j>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<j>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<m>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<m>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<f>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<f>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<d>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<d>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<e>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<e>, initial size: 300, update size: 180
randomCudaMemUpdateImpl<N14sparse_end2end6common8TestTypeE>, initial size: 100, update size: 200
randomCudaMemUpdateImpl<N14sparse_end2end6common8TestTypeE>, initial size: 300, update size: 180
[       OK ] CUDAwrapperUnitTest.CudaMemUpdateTest (5 ms)
[ RUN      ] CUDAwrapperUnitTest.CudaMemUpdateTwiceTest
randomCudaMemUpdateTwiceImpl<c>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<h>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<a>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<s>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<i>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<l>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<h>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<t>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<j>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<m>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<f>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<d>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<e>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<N14sparse_end2end6common8TestTypeE>, initial size: 100, second size: 200, third size: 300
randomCudaMemUpdateTwiceImpl<c>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<h>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<a>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<s>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<i>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<l>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<h>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<t>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<j>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<m>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<f>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<d>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<e>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<N14sparse_end2end6common8TestTypeE>, initial size: 100, second size: 300, third size: 200
randomCudaMemUpdateTwiceImpl<c>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<h>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<a>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<s>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<i>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<l>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<h>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<t>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<j>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<m>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<f>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<d>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<e>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<N14sparse_end2end6common8TestTypeE>, initial size: 200, second size: 300, third size: 100
randomCudaMemUpdateTwiceImpl<c>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<h>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<a>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<s>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<i>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<l>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<h>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<t>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<j>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<m>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<f>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<d>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<e>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<N14sparse_end2end6common8TestTypeE>, initial size: 200, second size: 100, third size: 300
randomCudaMemUpdateTwiceImpl<c>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<h>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<a>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<s>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<i>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<l>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<h>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<t>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<j>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<m>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<f>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<d>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<e>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<N14sparse_end2end6common8TestTypeE>, initial size: 300, second size: 200, third size: 100
randomCudaMemUpdateTwiceImpl<c>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<h>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<a>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<s>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<i>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<l>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<h>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<t>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<j>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<m>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<f>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<d>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<e>, initial size: 300, second size: 100, third size: 200
randomCudaMemUpdateTwiceImpl<N14sparse_end2end6common8TestTypeE>, initial size: 300, second size: 100, third size: 200
[       OK ] CUDAwrapperUnitTest.CudaMemUpdateTwiceTest (24 ms)
[----------] 7 tests from CUDAwrapperUnitTest (184 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 1 test suite ran. (184 ms total)
[  PASSED  ] 7 tests.
```