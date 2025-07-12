import torch
import numpy as np

# Parameters for the dummy grid
shape = (5, 5, 5)  # (Z, Y, X)
occ = torch.zeros(shape, dtype=torch.int32)

# Make a small dense block around (z=2, y=1, x=1)
occ[1:4, 0:3, 0:3] = 1  # Occupy a 3x3x3 block centered at (2,1,1)

# Save as occupancy.pt
torch.save(occ, "dummy_occupancy.pt")
print("Saved dummy_occupancy.pt with a dense 3x3x3 block around (z=2, y=1, x=1)")

# Print all occupied indices for verification
occupied = (occ > 0).nonzero()
print("Occupied indices (z, y, x):")
for idx in occupied:
    print(tuple(idx.tolist()))
