import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python check_npy_shape.py <path_to_npy_file>")
    sys.exit(1)

npy_path = sys.argv[1]
arr = np.load(npy_path)
print(f"File: {npy_path}")
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
if arr.ndim == 3:
    print(f"3D array: (dim0={arr.shape[0]}, dim1={arr.shape[1]}, dim2={arr.shape[2]})")
elif arr.ndim == 2:
    print(f"2D array: (dim0={arr.shape[0]}, dim1={arr.shape[1]})")
else:
    print(f"{arr.ndim}D array")
