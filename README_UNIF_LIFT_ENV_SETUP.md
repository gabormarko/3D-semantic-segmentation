## ENVIRONMENT:
unified_lift_cuda121   /home/neural_fields/miniconda3/envs/unified_lift_cuda121

(unified_lift_cuda121) neural_fields@VC04:~/Unified-Lift-Gabor$ python -c "import sys; import torch; print('Python version:', sys.version); print('Torch version:', torch.__version__); print('Torch CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available());" && gcc --version | head -n 1 | cut -d' ' -f3
Python version: 3.8.20 (default, Oct  3 2024, 15:24:27) 
[GCC 11.2.0]
Torch version: 2.4.1
Torch CUDA version: 12.1
CUDA available: True
11.4.0-1ubuntu1~22.04)

MOST IMPORTANT:
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

# Unified Lift Environment Setup (CUDA 12.1)

This guide provides step-by-step instructions to set up a working Python environment for Unified Lift with CUDA 12.1, including all necessary dependencies and submodules.

---

## 1. System Requirements
- Ubuntu 20.04/22.04 (or compatible Linux)
- NVIDIA GPU with CUDA 12.1 support
- CUDA 12.1 Toolkit installed at `/usr/local/cuda-12.1`
- `git`, `wget`, and `conda` (Miniconda/Anaconda) installed
- System development tools:
  ```bash
  sudo apt-get update
  sudo apt-get install build-essential libc6-dev
  ```

## 2. Clone the Repository and Submodules
```bash
git clone <your-repo-url> Unified-Lift-Gabor
cd Unified-Lift-Gabor
git submodule update --init --recursive
```

## 3. Create and Activate Conda Environment
```bash
conda create -n unified_lift_cuda121 python=3.8 -y
conda activate unified_lift_cuda121
```

## 4. Install PyTorch with CUDA 12.1
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

## 5. Set CUDA Environment Variables
Add these lines to your `~/.bashrc` (or run in your shell):
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
Then reload:
```bash
source ~/.bashrc
```

## 6. Install Python Dependencies
If a `requirements.txt` exists:
```bash
pip install -r requirements.txt
```
Or install packages as needed.

## 7. Build and Install Submodules
Before installing submodules, set the compiler to use the system GCC (important for CUDA builds in conda):
```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CONDA_BUILD_SYSROOT=/
```
Then build and install:
```bash
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## 8. (Optional) Install Other Submodules
If there are other submodules with `setup.py`, repeat:
```bash
pip install ./submodules/<submodule-name>
```

## 9. Verify Installation
Test imports in Python:
```python
import torch
import diff_gaussian_rasterization
import simple_knn
```

---

## Troubleshooting
- If you see errors about missing system libraries (e.g., `libpthread.so.0`), ensure you have installed `build-essential` and `libc6-dev`.
- If you see CUDA version mismatch errors, make sure both PyTorch and your system CUDA toolkit are version 12.1.
- If you use a different CUDA version, adjust the steps accordingly.
- If you encounter linker errors in a conda environment, make sure to set `export CC=/usr/bin/gcc`, `export CXX=/usr/bin/g++`, and `export CONDA_BUILD_SYSROOT=/` before building extensions.

---

## Notes
- Always activate your conda environment before running any Unified Lift scripts.
- For reproducibility, consider exporting your environment:
  ```bash
  conda env export > environment.yaml
  ```

---

This guide should allow you to fully reproduce a working Unified Lift environment with CUDA 12.1 support.
