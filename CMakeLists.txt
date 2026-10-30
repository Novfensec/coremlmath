cmake_minimum_required(VERSION 3.15)
project(coremlmath LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_cmlmath src/cpp/cmlmath.cpp)

if(ANDROID)
    target_compile_options(_cmlmath PRIVATE -fopenmp)
    target_link_libraries(_cmlmath PRIVATE -fopenmp -static-openmp)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(_cmlmath PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(_cmlmath PRIVATE /O2)
else()
    target_compile_options(_cmlmath PRIVATE -O3)
endif()

install(TARGETS _cmlmath DESTINATION .)