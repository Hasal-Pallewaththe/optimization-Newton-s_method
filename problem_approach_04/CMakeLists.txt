cmake_minimum_required (VERSION 3.0)
project (problem_approach_04)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable (Main ./src/main.cpp)
target_link_libraries (Main Eigen3::Eigen)