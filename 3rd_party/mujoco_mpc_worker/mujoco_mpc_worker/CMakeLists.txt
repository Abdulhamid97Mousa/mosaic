# CMakeLists.txt for MuJoCo MPC Worker
# This builds the MJPC binaries from the vendored mujoco_mpc source
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#
# The built binaries will be placed in build/bin/

cmake_minimum_required(VERSION 3.16)
project(mujoco_mpc_worker VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Path to vendored mujoco_mpc
set(MJPC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mujoco_mpc")

# Check if vendored source exists
if(NOT EXISTS "${MJPC_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Vendored mujoco_mpc not found at ${MJPC_SOURCE_DIR}\n"
        "Please ensure the mujoco_mpc submodule is initialized:\n"
        "  git submodule update --init --recursive"
    )
endif()

# Add the vendored mujoco_mpc as a subdirectory
# This will build all MJPC targets including the GUI and agent_server
add_subdirectory(${MJPC_SOURCE_DIR} ${CMAKE_BINARY_DIR}/mujoco_mpc)

# Create a custom target for easy reference
add_custom_target(mjpc_all
    DEPENDS mjpc agent_server
    COMMENT "Building all MJPC binaries"
)

# Print build info
message(STATUS "MuJoCo MPC Worker Build Configuration:")
message(STATUS "  MJPC Source: ${MJPC_SOURCE_DIR}")
message(STATUS "  Output Dir: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
