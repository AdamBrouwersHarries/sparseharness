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
include_directories("inc/executor")

set(EXECUTOR_SOURCE
	src/executor/util/Assert.cpp
	src/executor/util/Logger.cpp
	src/executor/Core.cpp
	src/executor/DeviceID.cpp
	src/executor/Event.cpp
	src/executor/JNIHandle.cpp
	src/executor/LocalArg.cpp
	src/executor/DeviceBuffer.cpp
	src/executor/DeviceList.cpp
	src/executor/Executor.cpp
	src/executor/KernelArg.cpp
	src/executor/PlatformID.cpp
	src/executor/ValueArg.cpp
	src/executor/Device.cpp
	src/executor/DeviceProperties.cpp
	src/executor/GlobalArg.cpp
	src/executor/Kernel.cpp
	src/executor/Source.cpp
	)


set(SPMV_SOURCE
    src/mmio.cpp
    src/kernel_config.cpp
    src/sparse_matrix.cpp
    # src/OpenCL_utils.cpp
    src/run.cpp
    src/arithexpr_evaluator.cpp
    src/csds_timer.cpp
    # src/kernel_with_args.cpp
    )

add_library (ExecutorLib ${EXECUTOR_SOURCE})

add_library (SpmvLib ${SPMV_SOURCE})

function(add_app name)
    add_executable(${name}_harness app/${name}.cpp)
    target_link_libraries(${name}_harness ExecutorLib SpmvLib 
    	${OpenCL_LIBRARIES} ${Boost_LIBRARIES})
endfunction()

add_app(spmv)
add_app(bfs)
