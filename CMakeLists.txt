cmake_minimum_required(VERSION 3.20)
project(oracle-llm-engine VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ORACLE_BUILD_PYTHON "Build pybind11 module" OFF)
option(ORACLE_BUILD_TESTS "Build unit tests" ON)
option(ORACLE_HAS_LLAMA_CPP "Link llama.cpp if present" OFF)

find_package(Accelerate REQUIRED)
find_library(METAL_LIBRARY Metal REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(QUARTZCORE_LIBRARY QuartzCore)

set(HTTPLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib.h")
if(NOT EXISTS "${HTTPLIB_PATH}")
  message(FATAL_ERROR "third_party/httplib.h missing; run scripts/fetch_deps.sh")
endif()

add_library(oracle_tb3
  src/tb3-transport/protocol.cpp
  src/tb3-transport/ring_buffer.cpp
  src/tb3-transport/socket_transport.cpp
)
target_include_directories(oracle_tb3 PUBLIC include)
target_compile_features(oracle_tb3 PUBLIC cxx_std_20)

add_library(oracle_shard
  src/mem-shard-manager/cluster_config.cpp
  src/mem-shard-manager/memory_shard_manager.cpp
  src/mem-shard-manager/kv_partitioner.cpp
  src/mem-shard-manager/pressure_monitor.cpp
)
target_include_directories(oracle_shard PUBLIC include)
target_link_libraries(oracle_shard PUBLIC oracle_tb3)

add_library(oracle_runner
  src/metal-cpu-runner/accelerate_runner.cpp
  src/metal-cpu-runner/llamacpp_runner.cpp
  src/metal-cpu-runner/metal_node_runner.mm
)
set_source_files_properties(src/metal-cpu-runner/metal_node_runner.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_include_directories(oracle_runner PUBLIC include)
target_compile_definitions(oracle_runner PRIVATE ACCELERATE_NEW_LAPACK=1)
target_link_libraries(oracle_runner PUBLIC oracle_shard Accelerate::Accelerate
  ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
if(QUARTZCORE_LIBRARY)
  target_link_libraries(oracle_runner PUBLIC ${QUARTZCORE_LIBRARY})
endif()
if(ORACLE_HAS_LLAMA_CPP)
  target_compile_definitions(oracle_runner PUBLIC ORACLE_HAS_LLAMA_CPP=1)
endif()

add_library(oracle_orch
  src/orchestrator-core/pipeline_orchestrator.cpp
  src/orchestrator-core/dag.cpp
  src/orchestrator-core/heartbeat.cpp
  src/orchestrator-core/openai_server.cpp
)
target_include_directories(oracle_orch PUBLIC include third_party)
target_link_libraries(oracle_orch PUBLIC oracle_runner oracle_tb3 oracle_shard)

add_executable(oracle-engine-master apps/oracle-engine-master.cpp)
target_link_libraries(oracle-engine-master PRIVATE oracle_orch)

add_executable(oracle-engine-worker apps/oracle-engine-worker.cpp)
target_link_libraries(oracle-engine-worker PRIVATE oracle_orch)

add_executable(oracle-engine-bench-net apps/oracle-engine-bench-net.cpp)
target_link_libraries(oracle-engine-bench-net PRIVATE oracle_tb3)

if(ORACLE_BUILD_TESTS)
  enable_testing()
  add_executable(test_ring_buffer tests/test_ring_buffer.cpp)
  target_link_libraries(test_ring_buffer PRIVATE oracle_tb3)
  add_test(NAME ring_buffer COMMAND test_ring_buffer)

  add_executable(test_protocol tests/test_protocol.cpp)
  target_link_libraries(test_protocol PRIVATE oracle_tb3)
  add_test(NAME protocol COMMAND test_protocol)

  add_executable(test_shard_plan tests/test_shard_plan.cpp)
  target_link_libraries(test_shard_plan PRIVATE oracle_shard)
  add_test(NAME shard_plan COMMAND test_shard_plan)

  add_executable(test_pipeline_tiny tests/test_pipeline_tiny.cpp)
  target_link_libraries(test_pipeline_tiny PRIVATE oracle_orch)
  add_test(NAME pipeline_tiny COMMAND test_pipeline_tiny)

  add_executable(test_transport_tcp tests/test_transport_tcp.cpp)
  target_link_libraries(test_transport_tcp PRIVATE oracle_tb3)
  add_test(NAME transport_tcp COMMAND test_transport_tcp)
endif()

if(ORACLE_BUILD_PYTHON)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6)
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(oracle_engine python/oracle_engine/bindings.cpp)
  target_link_libraries(oracle_engine PRIVATE oracle_orch)
endif()

install(TARGETS oracle-engine-master oracle-engine-worker oracle-engine-bench-net
        RUNTIME DESTINATION bin)
install(DIRECTORY include/oracle DESTINATION include)
