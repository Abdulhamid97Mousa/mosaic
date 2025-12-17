# How to Link Mosaic to MuJoCo MPC

## Prerequisites

Ensure you have the following system dependencies installed on Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    ninja-build \
    libgl1-mesa-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    zlib1g-dev \
    libstdc++-11-dev
```

## Python Package Installation

Install the Python worker package in editable mode:

```bash
# From the root of the repository
pip install -e 3rd_party/mujoco_mpc_worker
```

## Building the C++ Binary

The GUI requires the compiled `agent_server` binary to communicate with MuJoCo MPC.

1.  Navigate to the build directory:
    ```bash
    cd 3rd_party/mujoco_mpc_worker/mujoco_mpc/build
    ```

2.  Configure the build with CMake (enabling gRPC):
    ```bash
    # Clean previous build if necessary
    rm -rf * 
    
    cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON
    ```

3.  Compile the binaries:
    ```bash
    ninja -j$(nproc)
    ```

4.  Verify installation:
    Check that `bin/agent_server` exists in the build directory.

## Troubleshooting

### Application Opens and Closes Immediately

This is often caused by a segmentation fault in the default "Quadruped Flat" task.

**Diagnosis:**
Run the binary manually to see the error:
```bash
./bin/mjpc
```
If you see `Segmentation fault (core dumped)`, try running with a different task:
```bash
./bin/mjpc --task="Cartpole"
```

**Fix:**
The launcher has been updated to default to "Cartpole" instead of "Quadruped Flat".
If you need to change the default task, modify `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/launcher.py`.

### Missing Shared Libraries

```bash
# From the build directory
./bin/agent_server
```
