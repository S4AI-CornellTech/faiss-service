project(faiss_server)
cmake_minimum_required (VERSION 3.18)
cmake_policy(SET CMP0003 NEW)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(FAISS_ROOT "${FAISS_ROOT}" CACHE PATH "Path to faiss source or install prefix")
set(FAISS_LIB_DIR "${FAISS_LIB_DIR}" CACHE PATH "Path to FAISS library directory")
if (EXISTS "${FAISS_ROOT}")
    include_directories("${FAISS_ROOT}")
endif()

if (NOT FAISS_LIB_DIR AND EXISTS "${FAISS_ROOT}/build/faiss")
    set(FAISS_LIB_DIR "${FAISS_ROOT}/build/faiss")
endif()

find_library(FAISS_LIBRARY
    NAMES faiss
    PATHS
        "${FAISS_LIB_DIR}"
        "${FAISS_ROOT}/lib"
        "${FAISS_ROOT}/build/faiss"
    NO_DEFAULT_PATH
)

if (NOT FAISS_LIBRARY)
    find_library(FAISS_LIBRARY NAMES faiss)
endif()

if (NOT FAISS_LIBRARY)
    message(FATAL_ERROR "FAISS library not found. Set FAISS_LIB_DIR or FAISS_ROOT.")
endif()

# Find 3rd parties
find_package(gflags REQUIRED)
find_package(spdlog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(GRPC REQUIRED)
find_package(prometheus-cpp CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Faiss Server
add_executable(faiss_server
    main.cpp faiss_server.cpp metrics.cpp metrics_reporter.cpp
    protobuf/faiss.grpc.pb.cc protobuf/faiss.pb.cc)

target_link_libraries(faiss_server
    ${FAISS_LIBRARY} gomp mkl_intel_lp64 mkl_core mkl_sequential pthread m dl gflags spdlog::spdlog
    protobuf::libprotobuf prometheus-cpp::core prometheus-cpp::pull
    grpc++_unsecure gpr absl_synchronization absl_base
    CUDA::cudart CUDA::cublas CUDA::cublasLt)
