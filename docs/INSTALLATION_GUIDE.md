# Installation Guide - Editable Mode

## What is Editable Install (`pip install -e`)?

**Editable mode** (also called "development mode") installs your project as a **symbolic link** rather than copying files to site-packages. This means:

- ✅ **Live code changes** - Any changes you make to the code are immediately available without reinstalling
- ✅ **Import from anywhere** - You can `import gym_gui` from any Python script
- ✅ **Best for development** - Perfect when actively developing the project
- ✅ **Dependencies managed** - All dependencies from `pyproject.toml` are installed

## Installation Options

### 1. Minimal Install (GUI only)
```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL
pip install -e .
```
**Installs:** Core GUI + Qt + base dependencies only

---

### 2. Chat Panel with Local LLM Support (Recommended)
```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL
pip install -e ".[chat]"
```
**Installs:**
- Core GUI
- `requests` - HTTP client for API calls
- `huggingface_hub` - Model download and authentication
- `vllm>=0.6.0` - Local GPU inference server

**Use Case:** Enables the Chat panel with:
- OpenRouter cloud models (GPT-4, Claude, etc.)
- vLLM local models (Llama, Mistral, Qwen, etc.)

---

### 3. Multiple Extras
```bash
# Chat + PettingZoo multi-agent environments
pip install -e ".[chat,pettingzoo]"

# Chat + CleanRL training + PettingZoo
pip install -e ".[chat,cleanrl,pettingzoo]"

# Chat + All environment families
pip install -e ".[chat,pettingzoo,vizdoom,nethack,crafter,procgen]"
```

---

### 4. Full Installation (Everything)
```bash
pip install -e ".[full]"
```
**Installs:** All optional dependencies (all workers, all environments, all features)

⚠️ **Warning:** This is a large install (~10GB+) and requires CUDA for vLLM/PyTorch GPU support

---

## Available Extras

| Extra | Description | Key Dependencies |
|-------|-------------|------------------|
| `chat` | LLM Chat UI (OpenRouter + vLLM) | `vllm`, `huggingface_hub`, `requests` |
| `cleanrl` | CleanRL training algorithms | `torch`, `stable-baselines3`, `tensorboard` |
| `pettingzoo` | Multi-agent environments | `pettingzoo`, `supersuit` |
| `vizdoom` | ViZDoom FPS environments | `vizdoom` |
| `nethack` | NetHack/MiniHack roguelikes | `nle`, `minihack` |
| `crafter` | Crafter survival benchmark | `crafter` |
| `procgen` | Procgen benchmark (16 envs) | `procgen` or `procgen-mirror` |
| `ray-rllib` | Ray/RLlib distributed training | `ray[rllib]` |
| `xuance` | XuanCe MARL algorithms | `xuance` |
| `jason` | Jason BDI agents | Java + Jason runtime |
| `spade-bdi` | SPADE BDI agents | `spade`, `spade-bdi` |
| `full` | Everything | All of the above |

---

## Verifying Installation

### Check Installed Package
```bash
pip show gym-gui
```
Should show:
```
Name: gym-gui
Location: /home/hamid/Desktop/Projects/GUI_BDI_RL
Editable project location: /home/hamid/Desktop/Projects/GUI_BDI_RL
```

### Check vLLM is Available
```bash
python -c "import vllm; print(f'vLLM {vllm.__version__} installed')"
```

### Test Import
```bash
python -c "from gym_gui.services.llm import ModelManager; print('LLM imports OK')"
```

---

## Using vLLM for Local Models

### 1. Start vLLM Server Manually
```bash
# Start vLLM with a downloaded model
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.9
```

### 2. Or Use GUI Auto-Start
The GUI's Chat panel will automatically:
1. Check if model is downloaded
2. Download if missing (with HuggingFace token)
3. Start vLLM server when you select "Provider: Local"
4. Load the selected model

---

## Common Issues

### 1. vLLM Installation Fails
```bash
# vLLM requires CUDA. Check NVIDIA drivers:
nvidia-smi

# If no GPU or drivers, vLLM won't work
# Use OpenRouter cloud provider instead
```

### 2. CUDA Out of Memory
```bash
# Use smaller models or reduce GPU memory usage
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --gpu-memory-utilization 0.7
```

### 3. Model Download Fails (403 Forbidden)
```
Error: Access to model meta-llama/Llama-3.1-8B-Instruct is restricted
```

**Solution:**
1. Visit the model page: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Enter your HuggingFace token in the Chat panel
4. Retry download

---

## Uninstalling

### Remove Editable Install
```bash
pip uninstall gym-gui
```

### Keep Environment, Reinstall Fresh
```bash
pip uninstall gym-gui
pip install -e ".[chat]"
```

---

## Alternative: Using requirements.txt

If you prefer not to use editable mode:

```bash
# Standard installation (includes chat extras)
pip install -r requirements.txt

# Specific extras only
pip install -r requirements/chat.txt
pip install -r requirements/cleanrl_worker.txt
```

**Note:** This does NOT install the GUI package in editable mode. You'll need to reinstall after code changes.

---

## Development Workflow

```bash
# 1. Clone/pull latest code
cd /home/hamid/Desktop/Projects/GUI_BDI_RL
git pull

# 2. Install in editable mode with chat support
pip install -e ".[chat]"

# 3. Make changes to code
# Files in gym_gui/ are now live-linked

# 4. Test changes immediately
python run_gui.py

# No need to reinstall! Changes are live.
```

---

## Recommended Setup for Your Project

Based on your usage, I recommend:

```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL

# Install with chat support (includes vLLM)
pip install -e ".[chat]"

# This installs:
# - gym_gui package in editable mode
# - vllm for local model serving
# - huggingface_hub for model downloads
# - requests for API calls
```

Then you can:
- Edit code in `gym_gui/` and changes are live
- Use Chat panel with local models (Qwen, Llama, etc.)
- Download models via HuggingFace Hub
- Start vLLM server automatically from GUI
