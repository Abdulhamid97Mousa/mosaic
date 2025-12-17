# MuJoCo MPC Installation and Troubleshooting Guide

This guide details the steps to install, build, and troubleshoot MuJoCo MPC (MJPC) within the `GUI_BDI_RL` project structure.

## 1. System Prerequisites (Ubuntu)

Before cloning and building, ensure you have the necessary system dependencies installed. These are required for MuJoCo's rendering and simulation engine.

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    libgl1-mesa-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    ninja-build \
    zlib1g-dev \
    clang \
    git
```

## 2. Cloning the Repository

You need to clone the `mujoco_mpc` repository into the specific worker directory in your project.

```bash
# Navigate to the worker directory
cd 3rd_party/mujoco_mpc_worker

# Clone the repository (if not already present)
# If the directory 'mujoco_mpc' already exists and is empty, remove it first or clone into it.
git clone https://github.com/google-deepmind/mujoco_mpc.git
```

## 3. Building MuJoCo MPC

Once cloned, you need to build the project using CMake and Ninja.

```bash
# Navigate to the cloned directory
cd mujoco_mpc

# Create a build directory
mkdir build
cd build

# Configure the build
# We use Release build type for performance
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja

# Build the project
# -j$(nproc) uses all available CPU cores for faster compilation
ninja -j$(nproc)
```

## 4. Running the Application

After a successful build, the binary is located in `build/bin`.

```bash
cd bin
./mjpc
```

## 5. Troubleshooting: Segmentation Fault (Core Dumped)

If you encounter a `Segmentation fault (core dumped)` when running `./mjpc`, it is likely an issue with OpenGL rendering or GPU drivers.

### Common Causes & Fixes

#### A. Missing or Incompatible GPU Drivers
MuJoCo requires working OpenGL support. If you have an NVIDIA GPU, ensure the proprietary drivers are installed and loaded.

1.  **Check NVIDIA Driver Status:**
    ```bash
    nvidia-smi
    ```
    If this command fails or shows an error, you may need to install or reinstall your drivers.

2.  **Install Drivers (Ubuntu):**
    ```bash
    sudo ubuntu-drivers autoinstall
    sudo reboot
    ```

#### B. Missing OpenGL Libraries
Even with drivers, some OpenGL libraries might be missing.

```bash
sudo apt-get install libgl1-mesa-glx libgl1-mesa-dri
```

#### C. Forcing Software Rendering (Debugging)
If you suspect hardware rendering is the issue, you can try forcing software rendering to verify if the application runs (it will be slow).

```bash
export LIBGL_ALWAYS_SOFTWARE=1
./mjpc
```
*If this works, your issue is definitely with the GPU drivers.*

#### D. Wayland vs X11
If you are using Ubuntu 22.04 or later, you might be on Wayland. MuJoCo/GLFW sometimes has issues with Wayland. Try switching to X11 or forcing the backend:

```bash
# Try forcing X11 backend for GLFW
export GLFW_IM_MODULE=ibus
./mjpc
```

#### E. Verify Dependencies
Ensure all dependencies from Step 1 were installed successfully without errors.

## 6. Integration with GUI_BDI_RL

The `GUI_BDI_RL` application expects the `mjpc` binary to be available at:
`3rd_party/mujoco_mpc_worker/mujoco_mpc/build/bin/mjpc`

Ensure you have built it in exactly this location for the GUI launcher to find it.
