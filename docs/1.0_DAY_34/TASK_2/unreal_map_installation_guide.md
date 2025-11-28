# Unreal-MAP Installation Guide

**Date:** 2025-11-27
**Target:** Unreal Engine 4.27 + Unreal-MAP
**Approach:** Install standalone first, analyze, then integrate into Mosaic/gym_gui

---

## Important Clarification

### The Problem with UE4.27 on Linux

1. **Epic Games does NOT provide pre-built UE4.27 for Linux** - only UE5.x is available as pre-built
2. **Unreal-MAP requires UE 4.27** (modified version), not UE5
3. **`Setup.sh` only exists in SOURCE CODE** downloads, not in pre-built packages
4. **The Unreal-MAP authors provide their own modified UE4.27** via OneDrive

### Your Options

| Option | Difficulty | Description |
|--------|------------|-------------|
| **A. Docker Image** | Easy | Use pre-configured Docker with binaries |
| **B. Pre-built Binaries** | Medium | Download compiled binaries (if available) |
| **C. Build from Source** | Hard | Build modified UE4.27 from source (~2+ hours) |

---

## Option A: Docker Image (Recommended for Quick Start)

The Unreal-MAP developers provide a **Docker image** on Docker Hub that includes:
- HMAP framework
- Pre-compiled Unreal-MAP binaries
- Environment configurations

```bash
# Pull the Docker image (check hmp2g repo for exact image name)
docker pull [image-name-from-hmp2g-docs]

# Run training in container
docker run -it --gpus all [image-name] python main.py --cfg attack_post.jsonc
```

**Pros:** Works in minutes, no UE4 build required
**Cons:** Less flexibility, container overhead

---

## Option B: Pre-built Binaries (Linux Headless Training)

The project supports **headless training** where you only need the compiled server binary, not the full UE4 Editor.

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Training Mode (Headless - No Graphics)                     │
│  ─────────────────────────────────────                      │
│  Linux Server Binary (BuildLinuxServer.py output)           │
│       ↕ TCP/UDP Communication                               │
│  Python Training (hmp_minimal_modules/main.py)              │
├─────────────────────────────────────────────────────────────┤
│  Visualization Mode (Optional - Separate Machine)           │
│  ─────────────────────────────────────                      │
│  Windows/Linux Render Binary (BuildLinuxRender.py output)   │
│  Can run on different machine for cross-platform rendering  │
└─────────────────────────────────────────────────────────────┘
```

### Download Pre-built Binaries

Check the hmp2g repository for download scripts:
- Repository: https://github.com/binary-husky/hmp2g
- The `Please_Run_This_First_To_Fetch_Big_Files.py` script may download pre-built binaries

```bash
cd ~/Desktop/Projects/GUI_BDI_RL/3rd_party/unreal_map_worker/unreal-map

# This script downloads large files including possible pre-built binaries
python3 Please_Run_This_First_To_Fetch_Big_Files.py
```

---

## Option C: Build from Source (Full Installation)

If you need the full UE4 Editor or pre-built binaries aren't available.

---

## Case 1: Windows Installation (Build from Source)

### System Requirements

| Requirement | Specification |
|-------------|---------------|
| OS | Windows 10/11 64-bit |
| CPU | Quad-core Intel/AMD, 2.5 GHz+ |
| RAM | 16 GB minimum (32 GB recommended) |
| GPU | NVIDIA GTX 1060+ or AMD RX 580+ |
| Disk Space | **150+ GB** free (SSD recommended) |
| Visual Studio | 2019 or 2022 Community Edition |

### Step 1: Install Visual Studio

1. Download [Visual Studio Community](https://visualstudio.microsoft.com/)

2. During installation, select these workloads:
   - ✅ **Desktop development with C++**
   - ✅ **Game development with C++**

3. In "Individual Components", ensure these are selected:
   - ✅ .NET Framework 4.6.2 SDK
   - ✅ .NET Framework 4.6.2 targeting pack
   - ✅ Windows 10 SDK (latest)
   - ✅ MSVC v142 or v143 build tools

### Step 2: Download Modified UE4.27 Source

> **Critical:** Unreal-MAP uses a **MODIFIED version** of UE4.27, not the official Epic Games release. The `Setup.bat` and `GenerateProjectFiles.bat` scripts are included in THIS download.

Download from OneDrive (provided by Unreal-MAP authors):
```
https://ageasga-my.sharepoint.com/:u:/g/personal/fuqingxu_yiteam_tech/Ee3lQrUjKNFMjPITm5G-hEgBbeEN6dMOPtKP9ssgONKJcA?e=BavOoJ
```

### Step 3: Build Unreal Engine

1. **Extract** the source code to a drive with 150GB+ free space
   ```
   Example: D:\UnrealEngine\
   ```

2. **Run Setup.bat** (included in the download)
   - Double-click `Setup.bat`
   - This downloads engine binaries (3-4 GB)
   - Wait for completion (may take 10-30 minutes)

3. **Run GenerateProjectFiles.bat**
   - Double-click `GenerateProjectFiles.bat`
   - Takes less than 1 minute

4. **Build in Visual Studio**
   - Double-click `UE4.sln` to open in Visual Studio
   - Set configuration: **Development Editor** | **Win64**
   - Right-click **UE4** in Solution Explorer → **Build**
   - Wait 20-60 minutes (depending on CPU)

### Step 4: Clone and Setup Unreal-MAP Project

```powershell
# Clone the repository
git clone https://github.com/binary-husky/unreal-map.git

# Navigate to project
cd unreal-map

# Download large files (includes assets and possibly pre-built binaries)
python Please_Run_This_First_To_Fetch_Big_Files.py
```

### Step 5: Configure Unreal-MAP Project

1. Right-click `UHMP.uproject`
2. Select **"Switch Unreal Engine version"**
3. Choose **"Source build at D:\UnrealEngine"** (your UE4 path)
4. Open generated `UHMP.sln` in Visual Studio
5. Set configuration: **Development Editor** | **Win64**
6. Build the UHMP project

### Step 6: Run Unreal-MAP

Double-click `UHMP.uproject` to launch the Unreal Editor with Unreal-MAP loaded.

### Step 7: Run Python Training

```powershell
cd unreal-map\PythonExample\hmp_minimal_modules

# Install Python dependencies
pip install torch numpy cython

# Run training
python main.py --cfg attack_post.jsonc
```

---

## Case 2: Ubuntu 22.04 Installation (Build from Source)

### System Specifications (Reference)

```
Static hostname: hamidOnUbuntu
Operating System: Ubuntu 22.04.5 LTS
Kernel: Linux 6.8.0-87-generic
Architecture: x86-64
Hardware Vendor: ERYING
Hardware Model: Polestar Z790
```

### System Requirements

| Requirement | Specification |
|-------------|---------------|
| OS | Ubuntu 22.04 LTS 64-bit |
| CPU | Quad-core Intel/AMD, 2.5 GHz+ |
| RAM | 16 GB minimum (32 GB recommended) |
| GPU | NVIDIA with proprietary drivers |
| Disk Space | **150+ GB** free |

### Step 1: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Build essentials and compilers
sudo apt install -y build-essential clang cmake dos2unix git curl wget

# Mono (C# compiler for UE4 build tools)
sudo apt install -y mono-complete mono-mcs mono-devel mono-xbuild \
    mono-reference-assemblies-4.0 \
    libmono-system-data-datasetextensions4.0-cil \
    libmono-system-web-extensions4.0-cil \
    libmono-system-management4.0-cil \
    libmono-system-xml-linq4.0-cil \
    libmono-microsoft-build-tasks-v4.0-4.0-cil

# Graphics and UI libraries
sudo apt install -y libfreetype6-dev libgtk-3-dev libsdl2-dev \
    libgl1-mesa-dev libglu1-mesa-dev

# Multi-monitor support
sudo apt install -y libxinerama-dev libxrandr-dev x11proto-xinerama-dev

# Qt for dialogs
sudo apt install -y qtbase5-dev

# Utilities
sudo apt install -y xdg-user-dirs p7zip-full

# Python dependencies for training
sudo apt install -y python3-pip python3-dev cython3
```

**One-liner:**
```bash
sudo apt install -y build-essential clang cmake dos2unix git curl wget \
    mono-complete mono-mcs mono-devel mono-xbuild mono-reference-assemblies-4.0 \
    libmono-system-data-datasetextensions4.0-cil libmono-system-web-extensions4.0-cil \
    libmono-system-management4.0-cil libmono-system-xml-linq4.0-cil \
    libmono-microsoft-build-tasks-v4.0-4.0-cil libfreetype6-dev libgtk-3-dev \
    libsdl2-dev libgl1-mesa-dev libglu1-mesa-dev libxinerama-dev libxrandr-dev \
    x11proto-xinerama-dev qtbase5-dev xdg-user-dirs p7zip-full python3-pip python3-dev cython3
```

### Step 2: Install NVIDIA Drivers (If Using NVIDIA GPU)

```bash
# Check current driver
nvidia-smi

# If not installed, install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot
```

### Step 3: Download Modified UE4.27 Source

> **Critical:** You MUST download the MODIFIED UE4.27 from the Unreal-MAP authors' OneDrive link. The official Epic Games website does NOT have UE4.27 pre-built for Linux, and their GitHub source requires Epic Games account linking.

**Download Link:**
```
https://ageasga-my.sharepoint.com/:u:/g/personal/fuqingxu_yiteam_tech/Ee3lQrUjKNFMjPITm5G-hEgBbeEN6dMOPtKP9ssgONKJcA?e=BavOoJ
```

```bash
# Create directory
mkdir -p ~/UnrealEngine
cd ~/UnrealEngine

# Download via browser from the OneDrive link above
# Then extract (assuming downloaded as .7z or .zip)
7z x UnrealEngine-4.27-modified.7z
# OR
unzip UnrealEngine-4.27-modified.zip
```

### Step 4: Build Unreal Engine

> **Note:** `Setup.sh` and `GenerateProjectFiles.sh` are included in the modified UE4.27 download, NOT from Epic's website.

```bash
cd ~/UnrealEngine

# Make scripts executable
chmod +x Setup.sh GenerateProjectFiles.sh

# Run setup (downloads ~3-4 GB of binaries)
./Setup.sh

# Generate project files
./GenerateProjectFiles.sh

# Build (this takes 30 min - 2+ hours)
# Use -j flag to specify parallel jobs (e.g., number of CPU cores)
make -j$(nproc)
```

**Build Tips:**
- Use `make -j8` if you have 8 cores
- Monitor with `htop` in another terminal
- If build fails with memory errors, reduce parallel jobs: `make -j4`

### Step 5: Clone and Setup Unreal-MAP Project

```bash
# Navigate to your project directory
cd ~/Desktop/Projects/GUI_BDI_RL/3rd_party/unreal_map_worker

# The repo should already be cloned as 'unreal-map'
# If not:
git clone https://github.com/binary-husky/unreal-map.git

cd unreal-map

# Download large files (IMPORTANT - may include pre-built binaries)
python3 Please_Run_This_First_To_Fetch_Big_Files.py
```

### Step 6: Build Unreal-MAP Binaries (If Needed)

If pre-built binaries were not downloaded in Step 5:

```bash
cd ~/Desktop/Projects/GUI_BDI_RL/3rd_party/unreal_map_worker/unreal-map

# Build Linux Server (headless, for training)
python3 BuildLinuxServer.py

# Build Linux Render (with graphics, for visualization)
python3 BuildLinuxRender.py
```

### Step 7: Run Unreal-MAP Editor

```bash
# Run the Unreal Editor with UHMP project
~/UnrealEngine/Engine/Binaries/Linux/UE4Editor \
    ~/Desktop/Projects/GUI_BDI_RL/3rd_party/unreal_map_worker/unreal-map/UHMP.uproject
```

### Step 8: Run Python Training

```bash
cd ~/Desktop/Projects/GUI_BDI_RL/3rd_party/unreal_map_worker/unreal-map/PythonExample/hmp_minimal_modules

# Install Python dependencies
pip3 install torch numpy cython pyximport

# Run a sample training task
python3 main.py --cfg attack_post.jsonc
```

---

## Quick Start: Python Training Only (No UE4 Build)

If you just want to test the Python training framework without the full Unreal Engine:

```bash
cd ~/Desktop/Projects/GUI_BDI_RL/3rd_party/unreal_map_worker/unreal-map/PythonExample/hmp_minimal_modules

# Install dependencies
pip3 install torch numpy cython pyximport

# Check what's available
ls *.jsonc

# Try running (will fail if UE4 server not running, but shows the framework)
python3 main.py --cfg attack_post.jsonc
```

This will show you the training framework structure even if the UE4 server isn't available.

---

## Summary: What You Actually Need

### For Training Only (Headless):
1. Pre-built Linux Server binary (from `Please_Run_This_First_To_Fetch_Big_Files.py` or Docker)
2. Python dependencies: `torch`, `numpy`, `cython`
3. Run `python main.py --cfg <task>.jsonc`

### For Full Development (Editor + Training + Visualization):
1. Modified UE4.27 source from OneDrive link
2. Build UE4 with `Setup.sh`, `GenerateProjectFiles.sh`, `make`
3. Build UHMP project
4. Run training with visualization

---

## Our Integration Approach

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Install Standalone                                │
│  ─────────────────────────────                              │
│  • Download pre-built binaries OR build from source         │
│  • Run Python training to verify it works                   │
│  • Run their native UE4 UI (if available)                   │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Analyze Architecture                              │
│  ─────────────────────────────                              │
│  • Study task_runner.py for training loop                   │
│  • Understand config.py for configuration                   │
│  • Map communication protocol (Python ↔ UE4 Server)         │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Progressive Integration                           │
│  ─────────────────────────────                              │
│  • Create unreal_map_worker wrapper                         │
│  • Build Multi-Agent tab in gym_gui                         │
│  • Connect signals and handlers                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Setup.sh` not found | You downloaded from Epic's website, not the OneDrive link. Use the modified UE4.27 from Unreal-MAP authors. |
| `mono` not found | `sudo apt install mono-complete` |
| Clang version mismatch | Install clang-14: `sudo apt install clang-14` |
| Out of memory during build | Reduce parallel jobs: `make -j4` |
| GPU not detected | Install NVIDIA drivers: `sudo ubuntu-drivers autoinstall` |
| Permission denied | Don't build as root; use regular user |
| Missing .NET Framework | Windows: Install via Visual Studio Installer |

---

## References

- [Unreal-MAP GitHub](https://github.com/binary-husky/unreal-map)
- [HMAP Framework](https://github.com/binary-husky/hmp2g)
- [Unreal-MAP Paper (arXiv)](https://arxiv.org/html/2503.15947)
- [UE4 Linux Development](https://dev.epicgames.com/documentation/en-us/unreal-engine/linux-development-requirements-for-unreal-engine)
- [Building UE4 on Linux](https://michaeljcole.github.io/wiki.unrealengine.com/Building_On_Linux/)
