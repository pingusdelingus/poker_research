cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(poker_research)

# Point to local libtorch
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

# 1. Find Torch (Mac version)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 2. Add all source files (The glob ensures we get converter.cpp, game.cpp, etc.)
file(GLOB SOURCES "*.cpp")

# 3. Create the executable 'poker_bot' instead of 'm'
add_executable(poker_bot ${SOURCES})

# 4. Link Torch
target_link_libraries(poker_bot "${TORCH_LIBRARIES}")
set_property(TARGET poker_bot PROPERTY CXX_STANDARD 17)
