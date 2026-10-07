cmake_minimum_required(VERSION 3.24)
project(sim_cuda_cpp LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED IMPORTED_TARGET jsoncpp)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

add_library(sim_cuda_cpp STATIC sim_cuda_cpp.cu sim_cupy_cuda.cu)
target_include_directories(sim_cuda_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(sim_cuda_cpp PUBLIC cxx_std_17)
set_target_properties(sim_cuda_cpp PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(sim_cuda_cpp PUBLIC CUDA::cudart)

add_executable(simulator_cuda_cpp simulator_cuda_cpp_main.cpp)
target_include_directories(simulator_cuda_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_features(simulator_cuda_cpp PRIVATE cxx_std_17)
target_link_libraries(simulator_cuda_cpp PRIVATE sim_cuda_cpp PkgConfig::JSONCPP ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
    add_executable(sim_cuda_cpp_smoke tests/sim_cuda_cpp_smoke.cu)
    target_link_libraries(sim_cuda_cpp_smoke PRIVATE sim_cuda_cpp)
    add_test(NAME sim_cuda_cpp_smoke COMMAND sim_cuda_cpp_smoke)
    add_test(NAME simulator_cuda_cpp_tiny
        COMMAND simulator_cuda_cpp -c ${CMAKE_CURRENT_SOURCE_DIR}/tests/sim_cuda_cpp_tiny.json)
    add_test(NAME simulator_cuda_cpp_linear
        COMMAND simulator_cuda_cpp -c ${CMAKE_CURRENT_SOURCE_DIR}/tests/sim_cuda_cpp_linear.json)
endif()
