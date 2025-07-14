# How to Rebuild and Reinstall a CUDA Extension After Compilation Errors

If you encounter the following error when building a CUDA extension with PyTorch:

```
Traceback (most recent call last):
  File "/home/neural_fields/miniconda3/envs/cuda/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1808, in _run_ninja_build
    subprocess.run(
  ...
subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

The above exception was the direct cause of the following exception:
...
  File "/home/neural_fields/miniconda3/envs/cuda/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1824, in _run_ninja_build
    raise RuntimeError(message) from e
RuntimeError: Error compiling objects for extension
```

## Solution Checklist

### 0. Check Your Environment
Activate the correct conda environment and verify Python, PyTorch, CUDA, and GCC versions:

```bash
conda activate cuda
python -c "import sys; import torch; print('Python version:', sys.version); print('Torch version:', torch.__version__); print('Torch CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available());" && gcc --version | head -n 1 | cut -d' ' -f3
```

Expected output:
```
Python version: 3.8.20 ...
Torch version: 1.12.1
Torch CUDA version: 11.3
CUDA available: True
11.4.0-1ubuntu1~22.04)
```

### 1. Set CUDA Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda-11.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. Set Compiler to GCC-9

```bash
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
```

### 3. Build the Extension

```bash
python setup.py build
```

### 4. Install the Extension

```bash
python setup.py install
```

If you still encounter errors, try cleaning previous builds:

```bash
python setup.py clean --all
rm -rf build/
python setup.py build
python setup.py install
```

## Notes
- Always activate the correct conda environment before building.
- Make sure your CUDA, PyTorch, and GCC versions are compatible.
- If you use a different CUDA version, update `CUDA_HOME` accordingly.
- If you use a virtual environment, you may omit `--user` in the install command.

---

If you follow these steps, you should be able to rebuild and reinstall your CUDA extension successfully after code changes or build errors.
