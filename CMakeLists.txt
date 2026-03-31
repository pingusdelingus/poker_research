cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(poker_research)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES "*.cpp")

add_executable(poker_bot ${SOURCES})
target_link_libraries(poker_bot "${TORCH_LIBRARIES}")
