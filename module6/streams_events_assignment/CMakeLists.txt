cmake_minimum_required(VERSION 3.18)
project(cuda_streams_events_assignment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(streams_events main.cu)
set_target_properties(streams_events PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(streams_events PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)