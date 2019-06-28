cmake_minimum_required(VERSION 3.7)
project(cuBERT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

option(cuBERT_ENABLE_GPU "Enable GPU support" OFF)
option(cuBERT_ENABLE_MKL_SUPPORT "Enable Intel MKL support" OFF)
OPTION(USE_SWIG "Enable SWIG to generate Java API" ON)

find_package(Threads REQUIRED)

# OpenMP Support
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(USE_SWIG)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
endif()

include(cub)
include(protobuf)

set(cuBERT_EXTERNAL_LIBRARIES
        ${protobuf_STATIC_LIBRARIES})
set(cuBERT_EXTERNAL_DEPENDENCIES
        cub
        protobuf)

include_directories(
        # Source and generated code.
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        # External dependencies.
        ${cub_INCLUDE_DIR}
        ${PROTOBUF_INCLUDE_DIRS})

if(USE_SWIG)
  find_package(SWIG REQUIRED)
  find_package(Java REQUIRED)
  find_package(JNI REQUIRED)
  include(UseJava)
  include(UseSWIG)
  set(SWIG_CXX_EXTENSION "cxx")
  set(SWIG_EXTRA_LIBRARIES "")
  set(SWIG_JAVA_EXTRA_FILE_EXTENSIONS ".java" "JNI.java")
  set(SWIG_MODULE_JAVA_LANGUAGE "JAVA")
  set(SWIG_MODULE_JAVA_SWIG_LANGUAGE_FLAG "java")
  set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/java")
  include_directories(Java_INCLUDE_DIRS)
  include_directories(JNI_INCLUDE_DIRS)
  include_directories($ENV{JAVA_HOME}/include)
  if(WIN32)
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bert/windows/x86_64")
      include_directories($ENV{JAVA_HOME}/include/win32)
  elseif(APPLE)
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bert/osx/x86_64")
      include_directories($ENV{JAVA_HOME}/include/darwin)
  else()
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bert/linux/x86_64")
      include_directories($ENV{JAVA_HOME}/include/linux)
  endif()
endif(USE_SWIG)

# MKL Support
if(cuBERT_ENABLE_MKL_SUPPORT)
    include(mkl)
    add_definitions(-DHAVE_MKL)
    list(APPEND cuBERT_EXTERNAL_LIBRARIES ${mkl_SHARED_LIBRARIES})
    list(APPEND cuBERT_EXTERNAL_DEPENDENCIES mkl_copy_shared_to_destination)
    include_directories(${mkl_INCLUDE_DIRS})
endif(cuBERT_ENABLE_MKL_SUPPORT)

if(cuBERT_ENABLE_GPU)
    # minimum 9.0 in cuda version
    find_package(CUDA 9.0 REQUIRED)
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA not found.")
    endif()

    add_definitions(-DHAVE_CUDA)

    # use cmake internal CUDA_ARCH_NAME switch
    # e.g. CUDA_ARCH_NAME="Auto" will autodetect
    #      CUDA_ARCH_NAME="All"  will use all arches
    cuda_select_nvcc_arch_flags(NVCC_ARCH_FLAGS ${CUDA_ARCH_NAME})
    list(APPEND CUDA_NVCC_FLAGS ${NVCC_ARCH_FLAGS})
    message(STATUS "Using CUDA arch flags: ${NVCC_ARCH_FLAGS_readable}")

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math;-Xptxas -O3;-Xcompiler -O3)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero

    include_directories(${CUDA_INCLUDE_DIRS})
endif(cuBERT_ENABLE_GPU)

# Let's get to work!
include(tf_core_framework)

set(SOURCE_FILES
        src/cuBERT.cpp
        src/utf8proc/utf8proc.c
        src/cuBERT/common.cpp
        src/cuBERT/tokenization.cpp
        src/cuBERT/Bert.cpp
        src/cuBERT/BertM.cpp
        src/cuBERT/tensorflow/Graph.cpp
        src/cuBERT/op/GELU.cpp src/cuBERT/op/GELU.cu
        src/cuBERT/op/Dense.cpp
        src/cuBERT/op/Embedding.cpp src/cuBERT/op/Embedding.cu
        src/cuBERT/op/Softmax.cpp src/cuBERT/op/Softmax.cu
        src/cuBERT/op/LayerNorm.cpp src/cuBERT/op/LayerNorm.cu
        src/cuBERT/op_att/AttentionSelf.cpp
        src/cuBERT/op_att/AttentionMask.cpp src/cuBERT/op_att/AttentionMask.cu
        src/cuBERT/op_att/BatchMatMul.cpp
        src/cuBERT/op_att/Transformer.cpp
        src/cuBERT/op_bert/BertEmbeddings.cpp
        src/cuBERT/op_bert/BertPooler.cpp src/cuBERT/op_bert/BertPooler.cu
        src/cuBERT/op_out/AdditionalOutputLayer.cpp)


if (cuBERT_ENABLE_GPU)
    cuda_add_library(cuBERT SHARED ${SOURCE_FILES})
    set_target_properties(cuBERT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    cuda_add_cublas_to_target(cuBERT)
else ()
    add_library(cuBERT SHARED ${SOURCE_FILES})
endif ()
add_dependencies(cuBERT ${cuBERT_EXTERNAL_DEPENDENCIES})
target_link_libraries(cuBERT tf_protos_cc ${cuBERT_EXTERNAL_LIBRARIES})


if(USE_SWIG)
  set_property(SOURCE swig/cuBERT.i PROPERTY CPLUSPLUS ON)
  LIST(APPEND swig_options -package com.dtwave.aibox.alg.textanalysis.bert)
  set_property(SOURCE swig/cuBERT.i PROPERTY SWIG_FLAGS "${swig_options}")
  swig_add_module(cuBERT_swig java swig/cuBERT.i)
  swig_link_libraries(cuBERT_swig cuBERT)
  # needed to ensure Linux build does not have lib prefix specified twice, e.g. liblibcuBERT_swig
  set_target_properties(cuBERT_swig PROPERTIES PREFIX "")
  # needed in some versions of CMake for VS and MinGW builds to ensure output dll has lib prefix
  set_target_properties(cuBERT_swig PROPERTIES OUTPUT_NAME "libcuBERT_swig")
  if(WIN32)
    if(MINGW OR CYGWIN)
        add_custom_command(TARGET cuBERT_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PROJECT_SOURCE_DIR}/libcuBERT.dll" bert/windows/x86_64
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PROJECT_SOURCE_DIR}/libcuBERT_swig.dll" bert/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf cuBERTlib.jar com)
    else()
        add_custom_command(TARGET cuBERT_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND cp "${PROJECT_SOURCE_DIR}/Release/*.dll" bert/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf cuBERTlib.jar com)
    endif()
  elseif(APPLE)
    add_custom_command(TARGET cuBERT_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND cp "${PROJECT_SOURCE_DIR}/build/*.dylib" bert/osx/x86_64
            COMMAND cp "${PROJECT_SOURCE_DIR}/build/libcuBERT_swig.jnilib" bert/osx/x86_64/libcuBERT_swig.dylib
            COMMAND "${Java_JAR_EXECUTABLE}" -cf cuBERTlib.jar com)
  else()
    add_custom_command(TARGET cuBERT_swig POST_BUILD
	    COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
	    COMMAND cp "${PROJECT_SOURCE_DIR}/build/*.so" bert/linux/x86_64
	    COMMAND "${Java_JAR_EXECUTABLE}" -cf cuBERTlib.jar com)
  endif()
endif(USE_SWIG)


add_executable(cuBERT_benchmark benchmark/benchmark_cu.cpp)
target_link_libraries(cuBERT_benchmark cuBERT)

find_package(tensorflow)
if (tensorflow_FOUND)
    add_executable(tfBERT_benchmark benchmark/benchmark_tf.cpp)
    target_include_directories(tfBERT_benchmark PUBLIC ${tensorflow_INCLUDE_DIR})
    target_link_libraries(tfBERT_benchmark ${tensorflow_LIBRARIES})
endif ()

enable_testing()
configure_file(test_vocab.txt ${CMAKE_CURRENT_BINARY_DIR}/test_vocab.txt COPYONLY)
include(googletest)
add_executable(cuBERT_test test/unit_test.cpp
        test/cuBERT/common_test.cpp
        test/cuBERT/tokenization_test.cpp
        test/cuBERT/op/DenseTest.cpp
        test/cuBERT/op/LayerNormTest.cpp
        test/cuBERT/op/SoftmaxTest.cpp
        test/cuBERT/op/GELUTest.cpp
        test/cuBERT/op/EmbeddingTest.cpp
        test/cuBERT/op_att/BatchMatMulTest.cpp
        test/cuBERT/op_att/AttentionSelfTest.cpp
        test/cuBERT/op_att/TransformerTest.cpp
        test/cuBERT/op_att/AttentionMaskTest.cpp
        test/cuBERT/op_bert/BertEmbeddingsTest.cpp
        test/cuBERT/op_bert/BertPoolerTest.cpp
        test/cuBERT/op_out/AdditionalOutputLayerTest.cpp
        test/cuBERT/BertTest.cpp
        test/cuBERT/BertMTest.cpp
        test/cuBERT_test.cpp)
target_link_libraries(cuBERT_test cuBERT ${googletest_STATIC_LIBRARIES})
target_include_directories(cuBERT_test PUBLIC ${googletest_INCLUDE_DIRS})
add_test(UnitTest cuBERT_test)


install(TARGETS cuBERT
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
install(FILES ${CMAKE_SOURCE_DIR}/src/cuBERT.h DESTINATION include)

install(FILES ${mkl_SHARED_LIBRARIES} DESTINATION lib)
install(DIRECTORY ${mkl_INCLUDE_DIRS} DESTINATION .)
