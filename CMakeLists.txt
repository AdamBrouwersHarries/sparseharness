cmake_minimum_required(VERSION 2.8)
project(Executor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pthread -fPIC")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

# include_directories(common)
include_directories(include)

find_package(OpenCL)
find_package(Boost REQUIRED COMPONENTS program_options)

# Enable ExternalProject CMake module
include(ExternalProject)

include_directories("${source_dir}/inc")
include_directories("inc")

set(SOURCE_FILES
    src/mmio.cpp
    src/kernel.cpp
    src/sparse_matrix.cpp
    src/OpenCL_utils.cpp
    )

add_library (Executor ${SOURCE_FILES})

function(add_app name)
    add_executable(${name}_harness app/${name}.cpp)
    target_link_libraries(${name}_harness Executor pthread ${OpenCL_LIBRARIES} ${Boost_LIBRARIES})
endfunction()

# include_directories(app)

add_app(spmv)
