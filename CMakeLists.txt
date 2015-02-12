cmake_minimum_required(VERSION 2.8)
project(cneuralnetwork)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# For testing trianing data class
add_executable(training_data_example training_data.cpp training_data_example.cpp)

# data training utility
add_executable(train_data train_data.cpp training_data.cpp multilayer_perceptron.cpp neuron.cpp weight.cpp layer.cpp training_data.cpp)

# XOR training example
add_executable(xor_example xor_example.cpp multilayer_perceptron.cpp neuron.cpp weight.cpp layer.cpp training_data.cpp)

