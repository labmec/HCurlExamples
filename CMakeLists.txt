# @file HDiv/CMakeList.txt  -- First file to read for CMake

cmake_minimum_required(VERSION 3.8...3.15)

project (HCurlExamples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Finding the pz package
find_package(NeoPZ REQUIRED)
add_subdirectory(HelmholtzExample)
add_subdirectory(HCurlProjection)
add_subdirectory(Sphere)

