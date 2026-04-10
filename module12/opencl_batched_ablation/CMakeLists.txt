cmake_minimum_required(VERSION 3.10)
project(OpenCL_Batched_Ablation
        VERSION 1.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find OpenCL ──────────────────────────────────
find_package(OpenCL REQUIRED)

# ── Source files ─────────────────────────────────
set(SOURCES
    src/main.cpp
    src/cl_setup.cpp
    src/device_info.cpp
    src/ablation_engine.cpp)

# ── Executable ───────────────────────────────────
add_executable(ablation_engine ${SOURCES})

target_include_directories(ablation_engine
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS})

target_link_libraries(ablation_engine
    PRIVATE
    ${OpenCL_LIBRARIES})

# ── Copy kernels to build directory ──────────────
file(GLOB KERNEL_FILES
     "${CMAKE_SOURCE_DIR}/kernels/*.cl")
file(COPY ${KERNEL_FILES}
     DESTINATION
     "${CMAKE_BINARY_DIR}/kernels")

# ── Install target ───────────────────────────────
install(TARGETS ablation_engine
        RUNTIME DESTINATION bin)
install(DIRECTORY kernels/
        DESTINATION bin/kernels)
