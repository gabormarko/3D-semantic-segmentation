import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import os
import sys

import numpy as np
import torch

from omegaconf import OmegaConf
from tqdm.notebook import trange, tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import open3d as o3d
import matplotlib.pyplot as plt

from lib.datasets import load_dataset
from lib.datasets.dataset import initialize_data_loader
from lib.utils.cuda.raycast_image import *

from MinkowskiEngine import SparseTensor

# Import default config
config = OmegaConf.load('./config/default.yaml')

## Load Dataset
config.net_2d.model = 'LSegNet'
config.data.chunks_folder = '3D_eroded_subset'
config.data.batch_size = 4
config.net_2d.downsample_ratio = 1.0
config.net_2d.max_image_num = 4

DatasetClass = load_dataset(config.data.dataset)
data_loader = initialize_data_loader(DatasetClass, config=config, phase=config.train.train_phase,
                                     num_workers=config.data.num_workers, augment_data=False,
                                     shuffle=False, repeat=False, batch_size=config.data.batch_size)

dataloader_iterator = iter(data_loader)

# Debugging for more dimensions
repeat_num = 15

## Load example batch
batch = next(dataloader_iterator)
coords, colors, target, rgb_images, label_images, poses_2d, color_intrinsics, scene_names, *transform = batch
coords = coords.cuda()
colors = (colors.cuda() / 255.).repeat(1, repeat_num).contiguous()
color_intrinsics = color_intrinsics.cuda()
rgb_images = (rgb_images.permute(0, 1, 3, 4, 2).repeat(1, 1, 1, 1, repeat_num).cuda().contiguous() + 1.) / 2.
poses_2d = poses_2d.cuda()

# Convert to SparseTensor
sinput_features = SparseTensor(colors, coords, requires_grad=True)

# Load chunks and initialize raycaster
chunk_dim = (torch.Tensor(config.data.chunk_size) / data_loader.dataset.VOXEL_SIZE).int()

raycaster = RaycastInterpolateFeatures(dims3d=chunk_dim.int(),
                                       width=data_loader.dataset.depth_shape[1],
                                       height=data_loader.dataset.depth_shape[0],
                                       voxel_size=data_loader.dataset.VOXEL_SIZE,
                                       config=config).cuda()

# Do forward pass
projected_feats, index_image, vox_dist_weights, mapping_nums = raycaster(sinput_features, poses_2d, color_intrinsics)

# Calculate gradient - that's actually the colors
grad_image = (rgb_images.view(-1, rgb_images.shape[-1]) + projected_feats.view(-1, projected_feats.shape[-1]) * 10e-5).contiguous()

# Get backward
grad_voxels = torch.zeros((coords.shape[0], rgb_images.shape[-1]), device=coords.device).contiguous()
grad_voxels = RayCastInterpolateFunction.backward_debug(raycaster.occ3d, index_image, vox_dist_weights,
                                                        grad_image, grad_voxels, mapping_nums)

grad_indexes = torch.zeros((coords.shape[0], 1), device=coords.device).contiguous()
grad_index_image = index_image / index_image.max()
grad_indexes = RayCastInterpolateFunction.backward_debug(raycaster.occ3d, index_image, vox_dist_weights,
                                                        grad_image, grad_voxels, mapping_nums)

# Visualize grad and projected grad
fig, axs = plt.subplots(1, 3, figsize=(8, 4))
axs[0].imshow(grad_image.view(*rgb_images.shape)[0, 0, :, :, :3].detach().cpu())
axs[1].imshow(projected_feats.view(*rgb_images.shape)[0, 0, :, :, :3].detach().cpu())
axs[2].imshow(index_image.view(*rgb_images.shape[:-1], 8)[0, 0, :, :, 0].detach().cpu())

fig.suptitle('Grad image with backward results', fontsize=24)
plt.show()

visualize = False
if visualize:
    gradient_pcd = o3d.geometry.PointCloud()
    gradient_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy())
    gradient_pcd.colors = o3d.utility.Vector3dVector(grad_voxels[:, :3].cpu().numpy())

    mapping_num_pcd = o3d.geometry.PointCloud()
    mapping_num_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy() + [100., 0., 0.])
    mapping_colors = torch.matmul((mapping_nums / mapping_nums.max()).view(-1, 1),
                                  torch.ones([3], device=mapping_nums.device).view(1, -1))
    mapping_num_pcd.colors = o3d.utility.Vector3dVector(mapping_colors.cpu().numpy())

    grad_indexes_pcd = o3d.geometry.PointCloud()
    grad_indexes_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy() - [100., 0., 0.])
    grad_indexes_colors = torch.matmul(grad_indexes.view(-1, 1),
                                       torch.ones([3], device=grad_indexes.device).view(1, -1))
    grad_indexes_pcd.colors = o3d.utility.Vector3dVector(grad_indexes_colors.cpu().numpy())

    o3d.visualization.draw_geometries([gradient_pcd, mapping_num_pcd, grad_indexes_pcd],
                                      window_name=f'Gradients as colors for backward pass')
