## Unit Test for `parameters_parser_unit_test`

```bash
# 1st Step
cd onboard/preprocessor/unit_test/parameters_parser_unit_test/
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
-- Configuring done
-- Generating done
-- Build files have been written to: /home/thomasvowu/PublishRepos/SparseEnd2End/onboard/preprocessor/unit_test/parameters_parser_unit_test/build
```

```bash
# 2nd Step
cmake --build build
```

You will get log like this:
```bash
[  2%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
[  4%] Linking CXX static library ../../../lib/libgtest.a
[  4%] Built target gtest
[  6%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o
[  8%] Linking CXX static library ../../../lib/libgtest_main.a
[  8%] Built target gtest_main
[ 10%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/contrib/graphbuilder.cpp.o
[ 12%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/contrib/graphbuilderadapter.cpp.o
[ 14%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/binary.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/convert.cpp.o
[ 18%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/depthguard.cpp.o
[ 20%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/directives.cpp.o
[ 22%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emit.cpp.o
[ 24%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitfromevents.cpp.o
[ 26%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitter.cpp.o
[ 28%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitterstate.cpp.o
[ 30%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitterutils.cpp.o
[ 32%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/exceptions.cpp.o
[ 34%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/exp.cpp.o
[ 36%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/fptostring.cpp.o
[ 38%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/memory.cpp.o
[ 40%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/node.cpp.o
[ 42%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/node_data.cpp.o
[ 44%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/nodebuilder.cpp.o
[ 46%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/nodeevents.cpp.o
[ 48%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/null.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/ostream_wrapper.cpp.o
[ 52%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/parse.cpp.o
[ 54%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/parser.cpp.o
[ 56%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/regex_yaml.cpp.o
[ 58%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scanner.cpp.o
[ 60%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scanscalar.cpp.o
[ 62%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scantag.cpp.o
[ 64%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scantoken.cpp.o
[ 66%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/simplekey.cpp.o
[ 68%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/singledocparser.cpp.o
[ 70%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/stream.cpp.o
[ 72%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/tag.cpp.o
[ 74%] Linking CXX static library libyaml-cpp.a
[ 74%] Built target yaml-cpp
[ 76%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock.dir/src/gmock-all.cc.o
[ 78%] Linking CXX static library ../../../lib/libgmock.a
[ 78%] Built target gmock
[ 80%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
[ 82%] Linking CXX static library ../../../lib/libgmock_main.a
[ 82%] Built target gmock_main
[ 84%] Building CXX object CMakeFiles/parameters_parser_unit_test.bin.dir/parameters_parser_unit_test.cpp.o
[ 86%] Building CXX object CMakeFiles/parameters_parser_unit_test.bin.dir/home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/preprocessor/parameters_parser.cpp.o
[ 88%] Linking CXX executable bin/parameters_parser_unit_test.bin
[ 88%] Built target parameters_parser_unit_test.bin
[ 90%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-sandbox.dir/sandbox.cpp.o
[ 92%] Linking CXX executable bin/sandbox
[ 92%] Built target yaml-cpp-sandbox
[ 94%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-parse.dir/parse.cpp.o
[ 96%] Linking CXX executable bin/parse
[ 96%] Built target yaml-cpp-parse
[ 98%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-read.dir/read.cpp.o
[100%] Linking CXX executable bin/read
[100%] Built target yaml-cpp-read
```

```bash
# 3rd Step
./build/bin/parameters_parser_unit_test.bin
```

You will get log like this:
```bash
Running main() from /home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/third_party/googletest/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from ParseParamsUnitTest
[ RUN      ] ParseParamsUnitTest.ParseParamsFunctionCall
[Preprocessor Parameters Infos]:
num_cams = 6
raw_img_c = 3
raw_img_h = 900
raw_img_w = 1600
model_input_img_c = 3
model_input_img_h = 256
model_input_img_w = 704

[ModelCfg Parameters Infos]:
embedfeat_dims = 256
sparse4d_extract_feat_shape_lc : 
89760 | 256 | 
sparse4d_extract_feat_spatial_shapes_ld : 
64 | 176 | 32 | 88 | 16 | 44 | 8 | 22 | 
sparse4d_extract_feat_level_start_index : 
First 5 elements: 0 | 11264 | 14080 | 14784 | 14960 | 
Last 5 elements: 74624 | 74800 | 86064 | 88880 | 89584 | 
multiview_multiscale_deformable_attention_aggregation_path = /path/to/onboard/assets/trt_engine/deformableAttentionAggr.so

[Sparse4dExtractFeatEngine Parameters Infos]:
sparse4d_extract_feat_engine.engine_path = /path/to/onboard/assets/trt_engine/sparse4dbackbone.engine
sparse4d_extract_feat_engine. input_names: 
img | 
sparse4d_extract_feat_engine. output_names=
feature | 

[Sparse4dHead1stEngine Parameters Infos]:
sparse4d_head1st_engine.engine_path = /path/to/onboard/assets/trt_engine/sparse4dhead1st.engine
sparse4d_head1st_engine. input_names: 
feature | spatial_shapes | level_start_index | instance_feature | anchor | time_interval | image_wh | lidar2img | 
sparse4d_head1st_engine. output_names=
pred_instance_feature | pred_anchor | pred_class_score | pred_quality_score | 

[Sparse4dHead2ndEngine Parameters Infos]:
sparse4d_head2nd_engine.engine_path = /path/to/onboard/assets/trt_engine/sparse4dhead2nd.engine
sparse4d_head2nd_engine. input_names: 
feature | spatial_shapes | level_start_index | instance_feature | anchor | time_interval | temp_instance_feature | temp_anchor | mask | track_id | image_wh | lidar2img | 
sparse4d_head2nd_engine. output_names=
pred_instance_feature | pred_anchor | pred_class_score | pred_quality_score | 

[InstanceBank Parameters Infos]:
num_querys = 900
query_dims = 11
kmeans_anchors : 
First 5 elements: -2.63228 | -10.6078 | -1.58141 | 1.07631 | 0.453459 | 
Last 5 elements: -0.0431661 | 0.973795 | 0.00122133 | 0.00690015 | -0.0716457 | 
topk_querys = 600
max_time_interval = 2.000000
default_time_interval = 0.500000
confidence_decay = 0.600000

[Postprocessor Parameters Infos]:
post_process_out_nums = 300
post_process_threshold = 0.200000
[       OK ] ParseParamsUnitTest.ParseParamsFunctionCall (2 ms)
[----------] 1 test from ParseParamsUnitTest (2 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2 ms total)
[  PASSED  ] 1 test.
```

## Unit Test for `img_preprocessor_unit_test`

```bash
# 1st Step
cd onboard/preprocessor/unit_test/img_preprocessor_unit_test/
cmake -B build -S .
```

You will get log like this:
```bash
-- The CXX compiler identification is GNU 9.4.0
-- The CUDA compiler identification is NVIDIA 11.6.55
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
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
-- Build files have been written to: /home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/preprocessor/unit_test/img_preprocessor_unit_test/build
```

```bash
# 2nd Step
cmake --build build -j8
```

You will get log like this:
```bash
[  1%] Building CXX object CMakeFiles/img_preprocessor_cuda.dir/home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/preprocessor/img_preprocessor.cpp.o
[  5%] Building CUDA object CMakeFiles/img_preprocessor_cuda.dir/home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/preprocessor/img_aug_with_bilinearinterpolation_kernel.cu.o
[  5%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
[  7%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/contrib/graphbuilder.cpp.o
[  9%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/contrib/graphbuilderadapter.cpp.o
[ 11%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/binary.cpp.o
[ 13%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/convert.cpp.o
[ 15%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/depthguard.cpp.o
[ 16%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/directives.cpp.o
[ 18%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emit.cpp.o
[ 20%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitfromevents.cpp.o
[ 22%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitter.cpp.o
[ 24%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitterstate.cpp.o
[ 26%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/emitterutils.cpp.o
[ 28%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/exceptions.cpp.o
[ 30%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/exp.cpp.o
[ 32%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/fptostring.cpp.o
[ 33%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/memory.cpp.o
[ 35%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/node.cpp.o
[ 37%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/node_data.cpp.o
[ 39%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/nodebuilder.cpp.o
[ 41%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/nodeevents.cpp.o
[ 43%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/null.cpp.o
[ 45%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/ostream_wrapper.cpp.o
[ 47%] Linking CXX static library libimg_preprocessor_cuda.a
[ 47%] Built target img_preprocessor_cuda
[ 49%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/parse.cpp.o
[ 50%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/parser.cpp.o
[ 52%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/regex_yaml.cpp.o
[ 54%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scanner.cpp.o
[ 56%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scanscalar.cpp.o
[ 58%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scantag.cpp.o
[ 60%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/scantoken.cpp.o
[ 62%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/simplekey.cpp.o
[ 64%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/singledocparser.cpp.o
[ 66%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/stream.cpp.o
[ 67%] Building CXX object third_party/yaml-cpp/CMakeFiles/yaml-cpp.dir/src/tag.cpp.o
[ 69%] Linking CXX static library libyaml-cpp.a
[ 69%] Built target yaml-cpp
[ 71%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-sandbox.dir/sandbox.cpp.o
[ 75%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-read.dir/read.cpp.o
[ 75%] Building CXX object third_party/yaml-cpp/util/CMakeFiles/yaml-cpp-parse.dir/parse.cpp.o
[ 77%] Linking CXX executable bin/parse
[ 79%] Linking CXX executable bin/read
[ 81%] Linking CXX executable bin/sandbox
[ 81%] Built target yaml-cpp-parse
[ 81%] Built target yaml-cpp-read
[ 81%] Built target yaml-cpp-sandbox
[ 83%] Linking CXX static library ../../../lib/libgtest.a
[ 83%] Built target gtest
[ 84%] Building CXX object third_party/googletest/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o
[ 86%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock.dir/src/gmock-all.cc.o
[ 88%] Linking CXX static library ../../../lib/libgtest_main.a
[ 88%] Built target gtest_main
[ 90%] Linking CXX static library ../../../lib/libgmock.a
[ 90%] Built target gmock
[ 92%] Building CXX object third_party/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
[ 94%] Linking CXX static library ../../../lib/libgmock_main.a
[ 94%] Built target gmock_main
[ 96%] Building CXX object CMakeFiles/img_preprocessor_unit_test.bin.dir/img_preprocessor_unit_test.cpp.o
[ 98%] Building CXX object CMakeFiles/img_preprocessor_unit_test.bin.dir/home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/preprocessor/parameters_parser.cpp.o
[100%] Linking CXX executable bin/img_preprocessor_unit_test.bin
[100%] Built target img_preprocessor_unit_test.bin
```

```bash
# 3rd Step
./build/bin/img_preprocessor_unit_test.bin
```

You will get log like this:
```bash
Running main() from /home/thomasvonwu/PublishRepos/SparseEnd2End/onboard/third_party/googletest/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from ImgPreprocessorUnitTest
[ RUN      ] ImgPreprocessorUnitTest.ImgPreprocessorFP32
[Time Cost] : Image Preprocessor (CUDA float32) time = 0.163744[ms].
[MaxError] = 0.0131711
[Time Cost] : Image Preprocessor (CUDA float32) time = 0.161792[ms].
[MaxError] = 0.0133002
[Time Cost] : Image Preprocessor (CUDA float32) time = 0.161792[ms].
[MaxError] = 0.0131727
[       OK ] ImgPreprocessorUnitTest.ImgPreprocessorFP32 (340 ms)
[----------] 1 test from ImgPreprocessorUnitTest (340 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (340 ms total)
[  PASSED  ] 1 test.
```