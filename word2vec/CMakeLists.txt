cmake_minimum_required(VERSION 3.3)
project(word2vec)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(word2vec ${SOURCE_FILES})