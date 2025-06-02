import torch
import numpy as np

class StandaloneMinkowskiVoxelGrid:
    def __init__(self, points, colors, voxel_size, device):
        from utils.hash_grid import MinkowskiVoxelGrid
        self.grid = MinkowskiVoxelGrid(points, colors=colors, voxel_size=voxel_size, device=device)

    def get_voxel_centers(self):
        return self.grid.get_voxel_centers()

    def get_features(self):
        return self.grid.get_features()

    def __len__(self):
        return len(self.grid)
