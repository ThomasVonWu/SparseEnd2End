## Unit Test for `eigen_third_party_unit_test`

```bash
# 1st Step
cd onboard/head/unit_test/eigen_third_party_unit_test/
cmake -B build -S .
```

You will get log like this:
```bash
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The C compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Performing Test EIGEN_COMPILER_SUPPORT_CPP11
-- Performing Test EIGEN_COMPILER_SUPPORT_CPP11 - Success
-- Performing Test COMPILER_SUPPORT_std=cpp03
-- Performing Test COMPILER_SUPPORT_std=cpp03 - Success
-- Performing Test standard_math_library_linked_to_automatically
-- Performing Test standard_math_library_linked_to_automatically - Success
-- Standard libraries to link to explicitly: none
-- Performing Test COMPILER_SUPPORT_WERROR
-- Performing Test COMPILER_SUPPORT_WERROR - Success
-- Performing Test COMPILER_SUPPORT_pedantic
-- Performing Test COMPILER_SUPPORT_pedantic - Success
-- Performing Test COMPILER_SUPPORT_Wall
-- Performing Test COMPILER_SUPPORT_Wall - Success
-- Performing Test COMPILER_SUPPORT_Wextra
-- Performing Test COMPILER_SUPPORT_Wextra - Success
-- Performing Test COMPILER_SUPPORT_Wundef
-- Performing Test COMPILER_SUPPORT_Wundef - Success
-- Performing Test COMPILER_SUPPORT_Wcastalign
-- Performing Test COMPILER_SUPPORT_Wcastalign - Success
-- Performing Test COMPILER_SUPPORT_Wcharsubscripts
-- Performing Test COMPILER_SUPPORT_Wcharsubscripts - Success
-- Performing Test COMPILER_SUPPORT_Wnonvirtualdtor
-- Performing Test COMPILER_SUPPORT_Wnonvirtualdtor - Success
-- Performing Test COMPILER_SUPPORT_Wunusedlocaltypedefs
-- Performing Test COMPILER_SUPPORT_Wunusedlocaltypedefs - Success
-- Performing Test COMPILER_SUPPORT_Wpointerarith
-- Performing Test COMPILER_SUPPORT_Wpointerarith - Success
-- Performing Test COMPILER_SUPPORT_Wwritestrings
-- Performing Test COMPILER_SUPPORT_Wwritestrings - Success
-- Performing Test COMPILER_SUPPORT_Wformatsecurity
-- Performing Test COMPILER_SUPPORT_Wformatsecurity - Success
-- Performing Test COMPILER_SUPPORT_Wshorten64to32
-- Performing Test COMPILER_SUPPORT_Wshorten64to32 - Failed
-- Performing Test COMPILER_SUPPORT_Wlogicalop
-- Performing Test COMPILER_SUPPORT_Wlogicalop - Success
-- Performing Test COMPILER_SUPPORT_Wenumconversion
-- Performing Test COMPILER_SUPPORT_Wenumconversion - Failed
-- Performing Test COMPILER_SUPPORT_Wcpp11extensions
-- Performing Test COMPILER_SUPPORT_Wcpp11extensions - Failed
-- Performing Test COMPILER_SUPPORT_Wdoublepromotion
-- Performing Test COMPILER_SUPPORT_Wdoublepromotion - Success
-- Performing Test COMPILER_SUPPORT_Wshadow
-- Performing Test COMPILER_SUPPORT_Wshadow - Success
-- Performing Test COMPILER_SUPPORT_Wnopsabi
-- Performing Test COMPILER_SUPPORT_Wnopsabi - Success
-- Performing Test COMPILER_SUPPORT_Wnovariadicmacros
-- Performing Test COMPILER_SUPPORT_Wnovariadicmacros - Success
-- Performing Test COMPILER_SUPPORT_Wnolonglong
-- Performing Test COMPILER_SUPPORT_Wnolonglong - Success
-- Performing Test COMPILER_SUPPORT_fnochecknew
-- Performing Test COMPILER_SUPPORT_fnochecknew - Success
-- Performing Test COMPILER_SUPPORT_fnocommon
-- Performing Test COMPILER_SUPPORT_fnocommon - Success
-- Performing Test COMPILER_SUPPORT_fstrictaliasing
-- Performing Test COMPILER_SUPPORT_fstrictaliasing - Success
-- Performing Test COMPILER_SUPPORT_wd981
-- Performing Test COMPILER_SUPPORT_wd981 - Failed
-- Performing Test COMPILER_SUPPORT_wd2304
-- Performing Test COMPILER_SUPPORT_wd2304 - Failed
-- Performing Test COMPILER_SUPPORT_STRICTANSI
-- Performing Test COMPILER_SUPPORT_STRICTANSI - Failed
-- Performing Test COMPILER_SUPPORT_Qunusedarguments
-- Performing Test COMPILER_SUPPORT_Qunusedarguments - Failed
-- Performing Test COMPILER_SUPPORT_ansi
-- Performing Test COMPILER_SUPPORT_ansi - Success
-- Performing Test COMPILER_SUPPORT_OPENMP
-- Performing Test COMPILER_SUPPORT_OPENMP - Success
-- Found unsuitable Qt version "5.9.6" from /home/thomasvonwu/anaconda3/bin/qmake
-- The Fortran compiler identification is GNU 9.4.0
-- Found unsuitable Qt version "5.9.6" from /home/thomasvonwu/anaconda3/bin/qmake
-- Qt4 not found, so disabling the mandelbrot and opengl demos
-- 
-- Configured Eigen 3.4.1
-- 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/head/unit_test/eigen_third_party_unit_test/build
```

```bash
# 2nd Step
cmake --build build -j8
```

You will get log like this:
```bash
[  0%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
[  0%] Linking CXX static library ../../../lib/libgtest.a
[  0%] Built target gtest
[  0%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o
[  0%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock.dir/src/gmock-all.cc.o
[  0%] Linking CXX static library ../../../lib/libgtest_main.a
[  0%] Built target gtest_main
[  0%] Linking CXX static library ../../../lib/libgmock.a
[  0%] Built target gmock
[  0%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
[100%] Linking CXX static library ../../../lib/libgmock_main.a
[100%] Built target gmock_main
[100%] Building CXX object CMakeFiles/eigen.bin.dir/eigen_third_party_unit_test.cpp.o
[100%] Linking CXX executable bin/eigen.bin
[100%] Built target eigen.bin
```

```bash
# 3rd Step
./build/bin/eigen.bin
```

You will get log like this:
```bash
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Eigen3rdPartyUnitTest
[ RUN      ] Eigen3rdPartyUnitTest.EigenIncludeTest
[       OK ] Eigen3rdPartyUnitTest.EigenIncludeTest (0 ms)
[----------] 1 test from Eigen3rdPartyUnitTest (0 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (0 ms total)
[  PASSED  ] 1 test.
```

## Unit Test for `instance_bank_unit_test`

```bash
# 1st Step
cd onboard/head/unit_test/instance_bank_unit_test/
cmake -B build -S .
```

You will get log like this:
```bash
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The C compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Performing Test EIGEN_COMPILER_SUPPORT_CPP11
-- Performing Test EIGEN_COMPILER_SUPPORT_CPP11 - Success
-- Performing Test COMPILER_SUPPORT_std=cpp03
-- Performing Test COMPILER_SUPPORT_std=cpp03 - Success
-- Performing Test standard_math_library_linked_to_automatically
-- Performing Test standard_math_library_linked_to_automatically - Success
-- Standard libraries to link to explicitly: none
-- Performing Test COMPILER_SUPPORT_WERROR
-- Performing Test COMPILER_SUPPORT_WERROR - Success
-- Performing Test COMPILER_SUPPORT_pedantic
-- Performing Test COMPILER_SUPPORT_pedantic - Success
-- Performing Test COMPILER_SUPPORT_Wall
-- Performing Test COMPILER_SUPPORT_Wall - Success
-- Performing Test COMPILER_SUPPORT_Wextra
-- Performing Test COMPILER_SUPPORT_Wextra - Success
-- Performing Test COMPILER_SUPPORT_Wundef
-- Performing Test COMPILER_SUPPORT_Wundef - Success
-- Performing Test COMPILER_SUPPORT_Wcastalign
-- Performing Test COMPILER_SUPPORT_Wcastalign - Success
-- Performing Test COMPILER_SUPPORT_Wcharsubscripts
-- Performing Test COMPILER_SUPPORT_Wcharsubscripts - Success
-- Performing Test COMPILER_SUPPORT_Wnonvirtualdtor
-- Performing Test COMPILER_SUPPORT_Wnonvirtualdtor - Success
-- Performing Test COMPILER_SUPPORT_Wunusedlocaltypedefs
-- Performing Test COMPILER_SUPPORT_Wunusedlocaltypedefs - Success
-- Performing Test COMPILER_SUPPORT_Wpointerarith
-- Performing Test COMPILER_SUPPORT_Wpointerarith - Success
-- Performing Test COMPILER_SUPPORT_Wwritestrings
-- Performing Test COMPILER_SUPPORT_Wwritestrings - Success
-- Performing Test COMPILER_SUPPORT_Wformatsecurity
-- Performing Test COMPILER_SUPPORT_Wformatsecurity - Success
-- Performing Test COMPILER_SUPPORT_Wshorten64to32
-- Performing Test COMPILER_SUPPORT_Wshorten64to32 - Failed
-- Performing Test COMPILER_SUPPORT_Wlogicalop
-- Performing Test COMPILER_SUPPORT_Wlogicalop - Success
-- Performing Test COMPILER_SUPPORT_Wenumconversion
-- Performing Test COMPILER_SUPPORT_Wenumconversion - Failed
-- Performing Test COMPILER_SUPPORT_Wcpp11extensions
-- Performing Test COMPILER_SUPPORT_Wcpp11extensions - Failed
-- Performing Test COMPILER_SUPPORT_Wdoublepromotion
-- Performing Test COMPILER_SUPPORT_Wdoublepromotion - Success
-- Performing Test COMPILER_SUPPORT_Wshadow
-- Performing Test COMPILER_SUPPORT_Wshadow - Success
-- Performing Test COMPILER_SUPPORT_Wnopsabi
-- Performing Test COMPILER_SUPPORT_Wnopsabi - Success
-- Performing Test COMPILER_SUPPORT_Wnovariadicmacros
-- Performing Test COMPILER_SUPPORT_Wnovariadicmacros - Success
-- Performing Test COMPILER_SUPPORT_Wnolonglong
-- Performing Test COMPILER_SUPPORT_Wnolonglong - Success
-- Performing Test COMPILER_SUPPORT_fnochecknew
-- Performing Test COMPILER_SUPPORT_fnochecknew - Success
-- Performing Test COMPILER_SUPPORT_fnocommon
-- Performing Test COMPILER_SUPPORT_fnocommon - Success
-- Performing Test COMPILER_SUPPORT_fstrictaliasing
-- Performing Test COMPILER_SUPPORT_fstrictaliasing - Success
-- Performing Test COMPILER_SUPPORT_wd981
-- Performing Test COMPILER_SUPPORT_wd981 - Failed
-- Performing Test COMPILER_SUPPORT_wd2304
-- Performing Test COMPILER_SUPPORT_wd2304 - Failed
-- Performing Test COMPILER_SUPPORT_STRICTANSI
-- Performing Test COMPILER_SUPPORT_STRICTANSI - Failed
-- Performing Test COMPILER_SUPPORT_Qunusedarguments
-- Performing Test COMPILER_SUPPORT_Qunusedarguments - Failed
-- Performing Test COMPILER_SUPPORT_ansi
-- Performing Test COMPILER_SUPPORT_ansi - Success
-- Performing Test COMPILER_SUPPORT_OPENMP
-- Performing Test COMPILER_SUPPORT_OPENMP - Success
-- Found unsuitable Qt version "5.9.6" from /home/thomasvonwu/anaconda3/bin/qmake
-- The Fortran compiler identification is GNU 9.4.0
-- Found unsuitable Qt version "5.9.6" from /home/thomasvonwu/anaconda3/bin/qmake
-- Qt4 not found, so disabling the mandelbrot and opengl demos
-- 
-- Configured Eigen 3.4.1
-- 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/head/unit_test/instance_bank_unit_test/build
```


```bash
# 2nd Step
cmake --build build -j16
```

You will get log like this:
```bash
[  0%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
[  0%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/contrib/graphbuilder.cpp.o
[  0%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/contrib/graphbuilderadapter.cpp.o
[  0%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/binary.cpp.o
[  0%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/convert.cpp.o
[  0%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/depthguard.cpp.o
[  0%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/directives.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emit.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitfromevents.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitter.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitterstate.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitterutils.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/exceptions.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/exp.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/fptostring.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/memory.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/node.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/node_data.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/nodebuilder.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/nodeevents.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/null.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/ostream_wrapper.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/parse.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/parser.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scanner.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/regex_yaml.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scanscalar.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scantag.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scantoken.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/simplekey.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/singledocparser.cpp.o
[ 66%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/stream.cpp.o
[ 66%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/tag.cpp.o
[ 66%] Linking CXX static library libyaml-cpp.a
[ 66%] Built target yaml-cpp
[ 66%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-sandbox.dir/sandbox.cpp.o
[ 66%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-parse.dir/parse.cpp.o
[ 66%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-read.dir/read.cpp.o
[ 66%] Linking CXX executable bin/parse
[ 66%] Linking CXX executable bin/read
[ 66%] Built target yaml-cpp-parse
[ 83%] Linking CXX executable bin/sandbox
[ 83%] Built target yaml-cpp-read
[ 83%] Built target yaml-cpp-sandbox
[100%] Linking CXX static library ../../../lib/libgtest.a
[100%] Built target gtest
[100%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock.dir/src/gmock-all.cc.o
[100%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o
[100%] Linking CXX static library ../../../lib/libgtest_main.a
[100%] Built target gtest_main
[100%] Linking CXX static library ../../../lib/libgmock.a
[100%] Built target gmock
[100%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
[100%] Linking CXX static library ../../../lib/libgmock_main.a
[100%] Built target gmock_main
[100%] Building CXX object CMakeFiles/instance_bank_unit_test.bin.dir/instance_bank_unit_test.cpp.o
[100%] Building CXX object CMakeFiles/instance_bank_unit_test.bin.dir/home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/preprocessor/parameters_parser.cpp.o
[100%] Building CXX object CMakeFiles/instance_bank_unit_test.bin.dir/home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/head/instance_bank.cpp.o
[100%] Building CXX object CMakeFiles/instance_bank_unit_test.bin.dir/home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/head/utils.cpp.o
[100%] Linking CXX executable bin/instance_bank_unit_test.bin
[100%] Built target instance_bank_unit_test.bin
```


```bash
# 3rd Step
./build/bin/instance_bank_unit_test.bin
```

You will get log like this:
```bash
Running main() from /home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/third_party/googletest/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from InstanceBankUnitTest
[ RUN      ] InstanceBankUnitTest.InstanceBankCpuImplInputOutputConsistencyVerification

[INFO] Instance bank error and time costs statistic, frameid = 1
[CPU Test] Instance Bank Get() Time Cost = 0.000938 [ms]
ibank_got_instance_feature [MaxError] = 0
ibank_got_kmeans_anchor [MaxError] = 0
[CPU Test] Instance Bank Cache() Time Cost = 1.57664 [ms]
ibank_cached_temp_confidence [MaxError] = 5.96046e-08
ibank_cached_confidence [MaxError] = 5.96046e-08
ibank_cached_feature [MaxError] = 0
ibank_cached_anchor [MaxError] = 0
[CPU Test] Instance Bank GetTrackId() Time Cost = 0.015958 [ms]
ibank_updated_cur_trackid [MaxError] = 0
ibank_updated_temp_trackid [MaxError] = 0

[INFO] Instance bank error and time costs statistic, frameid = 2
[CPU Test] Instance Bank Get() Time Cost = 0.144949 [ms]
ibank_got_instance_feature [MaxError] = 0
ibank_got_kmeans_anchor [MaxError] = 0
ibank_got_cached_feature [MaxError] = 0
ibank_got_cached_anchor [MaxError] = 2.38419e-07
[CPU Test] Instance Bank Cache() Time Cost = 1.37703 [ms]
ibank_cached_temp_confidence [MaxError] = 2.98023e-08
ibank_cached_confidence [MaxError] = 2.98023e-08
ibank_cached_feature [MaxError] = 0
ibank_cached_anchor [MaxError] = 0
[CPU Test] Instance Bank GetTrackId() Time Cost = 0.01437 [ms]
ibank_updated_cur_trackid [MaxError] = 0
ibank_updated_temp_trackid [MaxError] = 0

[INFO] Instance bank error and time costs statistic, frameid = 3
[CPU Test] Instance Bank Get() Time Cost = 0.228518 [ms]
ibank_got_instance_feature [MaxError] = 0
ibank_got_kmeans_anchor [MaxError] = 0
ibank_got_cached_feature [MaxError] = 0
ibank_got_cached_anchor [MaxError] = 2.38419e-07
[CPU Test] Instance Bank Cache() Time Cost = 1.05715 [ms]
ibank_cached_temp_confidence [MaxError] = 5.96046e-08
ibank_cached_confidence [MaxError] = 5.96046e-08
ibank_cached_feature [MaxError] = 0
ibank_cached_anchor [MaxError] = 0
[CPU Test] Instance Bank GetTrackId() Time Cost = 0.007692 [ms]
ibank_updated_cur_trackid [MaxError] = 0
ibank_updated_temp_trackid [MaxError] = 0
[       OK ] InstanceBankUnitTest.InstanceBankCpuImplInputOutputConsistencyVerification (32 ms)
[----------] 1 test from InstanceBankUnitTest (32 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (32 ms total)
[  PASSED  ] 1 test.
```
***Attention: It's import to use double in replace of float32 as function parameter precesion. The error decreased by 1,000x, lowering from \$10^{-4}$ to \$10^{-7}$ for `ibank_got_cached_anchor`. The following functions need use double:***
```c++
InstanceBank::get(const double& timestamp, const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat)
```
IF you use float type `timestamp` and `global_to_lidar_mat` parameters, you will get log like this:
```bash
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from InstanceBankUnitTest
[ RUN      ] InstanceBankUnitTest.InstanceBankCpuImplInputOutputConsistencyVerification

[INFO] Instance bank error and time costs statistic, frameid = 1
[CPU Test] Instance Bank Get() Time Cost = 0.000802 [ms]
ibank_got_instance_feature [MaxError] = 0
ibank_got_kmeans_anchor [MaxError] = 0
[CPU Test] Instance Bank Cache() Time Cost = 1.64195 [ms]
ibank_cached_temp_confidence [MaxError] = 5.96046e-08
ibank_cached_confidence [MaxError] = 5.96046e-08
ibank_cached_feature [MaxError] = 0
ibank_cached_anchor [MaxError] = 0
[CPU Test] Instance Bank GetTrackId() Time Cost = 0.017175 [ms]
ibank_updated_cur_trackid [MaxError] = 0
ibank_updated_temp_trackid [MaxError] = 0

[INFO] Instance bank error and time costs statistic, frameid = 2
[CPU Test] Instance Bank Get() Time Cost = 0.1616 [ms]
ibank_got_instance_feature [MaxError] = 0
ibank_got_kmeans_anchor [MaxError] = 0
ibank_got_cached_feature [MaxError] = 0
ibank_got_cached_anchor [MaxError] = 0.0001297
[CPU Test] Instance Bank Cache() Time Cost = 0.883202 [ms]
ibank_cached_temp_confidence [MaxError] = 2.98023e-08
ibank_cached_confidence [MaxError] = 2.98023e-08
ibank_cached_feature [MaxError] = 0
ibank_cached_anchor [MaxError] = 0
[CPU Test] Instance Bank GetTrackId() Time Cost = 0.008513 [ms]
ibank_updated_cur_trackid [MaxError] = 0
ibank_updated_temp_trackid [MaxError] = 0

[INFO] Instance bank error and time costs statistic, frameid = 3
[CPU Test] Instance Bank Get() Time Cost = 0.148992 [ms]
ibank_got_instance_feature [MaxError] = 0
ibank_got_kmeans_anchor [MaxError] = 0
ibank_got_cached_feature [MaxError] = 0
ibank_got_cached_anchor [MaxError] = 0.000144958
[CPU Test] Instance Bank Cache() Time Cost = 0.885197 [ms]
ibank_cached_temp_confidence [MaxError] = 2.98023e-08
ibank_cached_confidence [MaxError] = 2.98023e-08
ibank_cached_feature [MaxError] = 0
ibank_cached_anchor [MaxError] = 0
[CPU Test] Instance Bank GetTrackId() Time Cost = 0.007374 [ms]
ibank_updated_cur_trackid [MaxError] = 0
ibank_updated_temp_trackid [MaxError] = 0
[       OK ] InstanceBankUnitTest.InstanceBankCpuImplInputOutputConsistencyVerification (30 ms)
[----------] 1 test from InstanceBankUnitTest (31 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (31 ms total)
[  ERROR  ] 1 test.
```

