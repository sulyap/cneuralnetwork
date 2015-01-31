cmake_minimum_required(VERSION 2.8)
project(cneuralnetwork)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(cneuralnetwork main.cpp multilayer_perceptron.cpp neuron.cpp weight.cpp layer.cpp)
