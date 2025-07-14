## ENVIRONMENT:
unified_lift  /home/neural_fields/miniconda3/envs/unified_lift


CHECK ENV:
(unified_lift) neural_fields@VC04:~/Unified-Lift-Gabor$ python -c "import sys; import torch; print('Python version:', sys.version); print('Torch version:', torch.__version__); print('Torch CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available());" && gcc --version | head -n 1 | cut -d' ' -f3
Python version: 3.8.20 (default, Oct  3 2024, 15:24:27) 
[GCC 11.2.0]
Torch version: 1.12.1
Torch CUDA version: 11.3
CUDA available: True
11.4.0-1ubuntu1~22.04)

MOST IMPORTANT:
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

# MinkowskiEngine Reproducible Installation Guide (CUDA 11.x, Ubuntu 20.04/22.04)

This guide provides a **fully reproducible process** for installing MinkowskiEngine with CUDA support, including all troubleshooting steps for compiler, CUDA, BLAS, and environment issues.

---

## 1. System Requirements

- **OS:** Ubuntu 20.04 or 22.04 (other Linux may work, but not tested)
- **GPU:** NVIDIA GPU with CUDA Compute Capability 6.0+
- **CUDA Toolkit:** 11.0 or 11.1 (matching your PyTorch version)
- **NVIDIA Driver:** Compatible with your CUDA version (run `nvidia-smi` to check)
- **Conda:** Miniconda or Anaconda installed
- **Git, wget, build tools:**  
  ```bash
  sudo apt-get update
  sudo apt-get install git wget build-essential
  ```

---

## 2. Install CUDA Toolkit 11.x

**Check if CUDA is already installed:**
```bash
ls /usr/local/ | grep cuda
```
If you see `cuda-11.0` or `cuda-11.1`, skip to the next step.  
If not, install CUDA 11.0 (example for Ubuntu 20.04):

```bash
# Add NVIDIA repo and key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

# Install toolkit only (not drivers)
sudo apt-get install -y cuda-toolkit-11-0
```

**Set environment variables (add to `~/.bashrc`):**
```bash
export CUDA_HOME=/usr/local/cuda-11.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
Then run:
```bash
source ~/.bashrc
```

**Verify:**
```bash
nvcc --version
```

---

## 3. Install GCC-9 and G++-9

MinkowskiEngine does **not** support GCC > 9 for CUDA 11.x.

```bash
sudo apt-get install gcc-9 g++-9
```

---

## 4. Create a Clean Conda Environment

```bash
conda create -n minkowski_cuda110 python=3.7 -y
conda activate minkowski_cuda110
```

---

## 5. Install PyTorch and Dependencies

**Install PyTorch 1.7.1 with CUDA 11.0:**
```bash
conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch -y
```

**Install build tools and BLAS:**
```bash
conda install openblas-devel cmake make git -c conda-forge -y
```

---

## 6. Clone MinkowskiEngine

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
```

---

## 7. Build and Install MinkowskiEngine

**Set compilers to GCC-9:**
```bash
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
```

**Install with custom BLAS (OpenBLAS):**
```bash
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

---

## 8. Troubleshooting

- **GCC version error:**  
  Make sure you set `CC` and `CXX` to GCC-9/G++-9 before building.
- **nvcc not found:**  
  Make sure `CUDA_HOME` is set to a valid CUDA 11.x path and `nvcc` is present.
- **BLAS errors:**  
  Ensure `openblas-devel` is installed in your conda environment.
- **Disk space errors:**  
  Free up space by deleting large files or unused conda environments.
- **Compiler not found or wrong environment:**  
  Always activate your conda environment and set compilers before building.

---

## 9. Verify Installation

```python
import MinkowskiEngine as ME
print(ME.__version__)
```

---

## 10. Clean Up

- Remove any temporary build directories if needed:
  ```bash
  rm -rf build
  ```

---

## 11. Reproducibility Tips

- Always use a **fresh conda environment**.
- Always set `CC` and `CXX` to GCC-9 before building.
- Match your CUDA and PyTorch versions.
- Export your environment for future use:
  ```bash
  conda env export > environment.yaml
  ```

---

## Example: One-Liner Setup (after CUDA and GCC-9 are installed)

```bash
conda create -n minkowski_cuda110 python=3.7 -y
conda activate minkowski_cuda110
conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch -y
conda install openblas-devel cmake make git -c conda-forge -y
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

---

**If you follow this guide exactly, you will have a reproducible, working MinkowskiEngine installation with CUDA support.**  
If you hit any error, check the troubleshooting section or ask for help with the exact error message.
