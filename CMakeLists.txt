cmake_minimum_required(VERSION 2.8)
project(Executor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall -Wextra -Wno-implicit-fallthrough -Wno-unused-parameter -fPIC -DDEBUG")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

# include_directories(common)
include_directories(include)

find_package(OpenCL)
find_package(Boost REQUIRED COMPONENTS program_options)

# Enable ExternalProject CMake module
include(ExternalProject)

include_directories("${source_dir}/inc")
include_directories("inc")
include_directories("inc/util")
include_directories("/usr/local/include")

set(UTIL_SOURCE
    src/util/Logger.cpp
    )

set(SPMV_SOURCE
    src/mmio.cpp
    src/kernel_config.cpp
    src/sparse_matrix.cpp
    src/run.cpp
    src/arithexpr_evaluator.cpp
    src/csds_timer.cpp
    )

add_library (UtilLib ${UTIL_SOURCE})

add_library (SpmvLib ${SPMV_SOURCE})

function(add_app name)
    add_executable(${name}_harness app/${name}.cpp)
    target_link_libraries(${name}_harness UtilLib SpmvLib 
        ${OpenCL_LIBRARIES} ${Boost_LIBRARIES})
        # ${OpenCL_LIBRARIES})
    	
endfunction()

add_app(spmv)
add_app(bfs)
add_app(sssp)
add_app(pr)
add_app(eigenvector)
