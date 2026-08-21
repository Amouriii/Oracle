cmake_minimum_required(VERSION 3.20)

# OBJCXX only exists on Apple toolchains; enabling it unconditionally makes the
# whole project unconfigurable on Linux (and therefore in Docker).
if(APPLE)
  project(oracle-llm-engine VERSION 0.2.0 LANGUAGES C CXX OBJCXX)
else()
  project(oracle-llm-engine VERSION 0.2.0 LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(APPLE)
  set(CMAKE_OBJCXX_STANDARD 20)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ORACLE_BUILD_PYTHON "Build pybind11 module" OFF)
option(ORACLE_BUILD_TESTS "Build unit tests" ON)
option(ORACLE_HAS_LLAMA_CPP "Link llama.cpp if present" OFF)
option(ORACLE_NATIVE_ARCH "Compile with -march=native" OFF)

find_package(Threads REQUIRED)

set(ORACLE_PLATFORM_LIBS Threads::Threads)
if(NOT APPLE AND UNIX)
  list(APPEND ORACLE_PLATFORM_LIBS rt)
endif()

set(HTTPLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib.h")
if(NOT EXISTS "${HTTPLIB_PATH}")
  message(FATAL_ERROR "third_party/httplib.h missing; run scripts/fetch_deps.sh")
endif()

# ---------------------------------------------------------------- compute -----
add_library(oracle_compute src/compute/blas.cpp)
target_include_directories(oracle_compute PUBLIC include)
target_compile_features(oracle_compute PUBLIC cxx_std_20)
target_link_libraries(oracle_compute PUBLIC ${ORACLE_PLATFORM_LIBS})
if(APPLE)
  find_package(Accelerate)
  if(Accelerate_FOUND)
    target_compile_definitions(oracle_compute PRIVATE ORACLE_HAS_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
    target_link_libraries(oracle_compute PUBLIC Accelerate::Accelerate)
  endif()
endif()
if(ORACLE_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" ORACLE_HAS_MARCH_NATIVE)
  if(ORACLE_HAS_MARCH_NATIVE)
    target_compile_options(oracle_compute PRIVATE -march=native)
  endif()
endif()

# -------------------------------------------------------------- transport -----
add_library(oracle_tb3
  src/tb3-transport/protocol.cpp
  src/tb3-transport/ring_buffer.cpp
  src/tb3-transport/socket_transport.cpp
)
target_include_directories(oracle_tb3 PUBLIC include)
target_compile_features(oracle_tb3 PUBLIC cxx_std_20)
target_link_libraries(oracle_tb3 PUBLIC ${ORACLE_PLATFORM_LIBS})

# ----------------------------------------------------------------- shard ------
add_library(oracle_shard
  src/mem-shard-manager/cluster_config.cpp
  src/mem-shard-manager/memory_shard_manager.cpp
  src/mem-shard-manager/kv_partitioner.cpp
  src/mem-shard-manager/pressure_monitor.cpp
)
target_include_directories(oracle_shard PUBLIC include)
target_link_libraries(oracle_shard PUBLIC oracle_tb3)

# ---------------------------------------------------------------- model -------
add_library(oracle_model
  src/model/gguf.cpp
  src/model/quant.cpp
  src/model/tokenizer.cpp
)
target_include_directories(oracle_model PUBLIC include)
target_link_libraries(oracle_model PUBLIC oracle_compute oracle_shard)

# --------------------------------------------------------------- runner -------
set(ORACLE_RUNNER_SOURCES
  src/metal-cpu-runner/accelerate_runner.cpp
  src/metal-cpu-runner/llamacpp_runner.cpp
  src/metal-cpu-runner/gguf_runner.cpp
)
if(APPLE)
  list(APPEND ORACLE_RUNNER_SOURCES src/metal-cpu-runner/metal_node_runner.mm)
else()
  list(APPEND ORACLE_RUNNER_SOURCES src/metal-cpu-runner/metal_stub.cpp)
endif()

add_library(oracle_runner ${ORACLE_RUNNER_SOURCES})
target_include_directories(oracle_runner PUBLIC include)
target_link_libraries(oracle_runner PUBLIC oracle_shard oracle_compute oracle_model)
if(APPLE)
  set_source_files_properties(src/metal-cpu-runner/metal_node_runner.mm
                              PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  find_library(METAL_LIBRARY Metal)
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(QUARTZCORE_LIBRARY QuartzCore)
  if(METAL_LIBRARY AND FOUNDATION_LIBRARY)
    target_link_libraries(oracle_runner PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
  endif()
  if(QUARTZCORE_LIBRARY)
    target_link_libraries(oracle_runner PUBLIC ${QUARTZCORE_LIBRARY})
  endif()
endif()
if(ORACLE_HAS_LLAMA_CPP)
  target_compile_definitions(oracle_runner PUBLIC ORACLE_HAS_LLAMA_CPP=1)
endif()

# -------------------------------------------------------------- security ------
add_library(oracle_security
  src/security/api_keys.cpp
  src/security/rate_limiter.cpp
  src/security/integrity.cpp
  src/security/audit_log.cpp
  src/security/gate.cpp
)
target_include_directories(oracle_security PUBLIC include)
target_link_libraries(oracle_security PUBLIC oracle_tb3)

# ----------------------------------------------------------- orchestrator -----
add_library(oracle_orch
  src/orchestrator-core/pipeline_orchestrator.cpp
  src/orchestrator-core/openai_server.cpp
  src/orchestrator-core/scheduler.cpp
  src/orchestrator-core/worker_registry.cpp
  src/orchestrator-core/dashboard.cpp
)
target_include_directories(oracle_orch PUBLIC include third_party)
target_link_libraries(oracle_orch PUBLIC oracle_runner oracle_tb3 oracle_shard oracle_security oracle_model)

# ------------------------------------------------------------------ apps ------
add_executable(oracle-engine-master apps/oracle-engine-master.cpp)
target_link_libraries(oracle-engine-master PRIVATE oracle_orch)

add_executable(oracle-engine-worker apps/oracle-engine-worker.cpp)
target_link_libraries(oracle-engine-worker PRIVATE oracle_orch)

add_executable(oracle-engine-bench-net apps/oracle-engine-bench-net.cpp)
target_link_libraries(oracle-engine-bench-net PRIVATE oracle_tb3)

add_executable(oracle-model-info apps/oracle-model-info.cpp)
target_link_libraries(oracle-model-info PRIVATE oracle_model oracle_runner)

# ----------------------------------------------------------------- tests ------
if(ORACLE_BUILD_TESTS)
  enable_testing()
  foreach(t ring_buffer protocol shard_plan pipeline_tiny transport_tcp gguf quant tokenizer
            security scheduler)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_include_directories(test_${t} PRIVATE tests)
    add_test(NAME ${t} COMMAND test_${t})
    set_tests_properties(${t} PROPERTIES TIMEOUT 120)
  endforeach()
  target_link_libraries(test_ring_buffer PRIVATE oracle_tb3)
  target_link_libraries(test_protocol PRIVATE oracle_tb3)
  target_link_libraries(test_shard_plan PRIVATE oracle_shard)
  target_link_libraries(test_pipeline_tiny PRIVATE oracle_orch)
  target_link_libraries(test_transport_tcp PRIVATE oracle_tb3)
  target_link_libraries(test_gguf PRIVATE oracle_model oracle_runner)
  target_link_libraries(test_quant PRIVATE oracle_model)
  target_link_libraries(test_tokenizer PRIVATE oracle_model)
  target_link_libraries(test_security PRIVATE oracle_security)
  target_link_libraries(test_scheduler PRIVATE oracle_orch)
endif()

# ---------------------------------------------------------------- python ------
if(ORACLE_BUILD_PYTHON)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6)
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(oracle_engine python/oracle_engine/bindings.cpp)
  target_link_libraries(oracle_engine PRIVATE oracle_orch)
endif()

install(TARGETS oracle-engine-master oracle-engine-worker oracle-engine-bench-net oracle-model-info
        RUNTIME DESTINATION bin)
install(DIRECTORY include/oracle DESTINATION include)
install(DIRECTORY configs/ DESTINATION share/oracle/configs)
