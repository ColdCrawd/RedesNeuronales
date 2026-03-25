cmake_minimum_required(VERSION 3.16)
project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)

set(ORT_ROOT) "/opt/onnxruntime" CACHE PATH "ONNX Runtime installation folder")

add_executable(inference main.cpp)

target_include_directories(inference PRIVATE
    ${ORT_ROOT}/include
    ${ORT_ROOT}/include/onnxruntime/core/session
}

target_link_directories(inference PRIVATE
    ${ORT_ROOT}/lib
}

target_link_libraries(inference PRIVATE onnxruntime)
