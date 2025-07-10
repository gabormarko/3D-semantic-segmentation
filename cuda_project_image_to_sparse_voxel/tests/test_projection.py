import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import torch

from omegaconf import OmegaConf
from tqdm.notebook import trange, tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

from lib.datasets.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.utils.pc_utils import read_plyfile, save_point_cloud
from lib.utils.utils import read_txt, fast_hist, per_class_iu, fast_hist_torch
from lib.constants.scannet_constants import *
from lib.models import load_model
from lib.models.encoders_2d import load_2d_model
from lib.datasets import load_dataset
from lib.datasets.dataset import initialize_data_loader

from MinkowskiEngine import SparseTensor
import open3d as o3d

from lib.utils.cuda.raycast_image import Project2DFeaturesCUDA

#Import default config
config = OmegaConf.load('config/default.yaml')

## Load dataset
config.net_2d.model = 'LSegNet'
config.data.chunks_folder = '3D_eroded_subset'
config.data.batch_size = 2
config.net_2d.downsample_ratio = 1.
config.net_2d.max_image_num = 2
config.augmentation.elastic_distortion = False

DatasetClass = load_dataset(config.data.dataset)
data_loader = initialize_data_loader(DatasetClass, config=config, phase=config.train.train_phase,
                                            num_workers=config.data.num_workers, augment_data=True,
                                            shuffle=False, repeat=False, batch_size=config.data.batch_size)

dataloader_iterator = iter(data_loader)


## Load example batch
batch = next(dataloader_iterator)
coords, colors, target, mappings, rgb_images, label_images, poses_2d, color_intrinsics, scene_names, *transform = batch
coords = coords.cuda()
colors = colors.cuda()
color_intrinsics = color_intrinsics.cuda()
rgb_images = (rgb_images.cuda() + 1.) / 2.
poses_2d = poses_2d.cuda()

# Preprocess input
colors = colors[:, :3].float()
colors[:, :3] = colors[:, :3] / 255.
sinput = SparseTensor(colors, coords)

# We need the batch ids to get the correct mapping index
batch_ids = sinput.C[:, 0]
voxel_num = batch_ids.shape[0]
frame_num = mappings.shape[1]
batch_ids = batch_ids.unsqueeze(1).expand(voxel_num, frame_num).unsqueeze(-1).long()  # broadcast to mappings

# Use colors instead of features for projection
encoded_images = rgb_images.permute(0,1,3,4,2)
encoded_images.shape, mappings.shape


## Load chunks and initialize raycaster
chunk_dim = (torch.Tensor([1.5, 1.5, 3.]) / data_loader.dataset.VOXEL_SIZE).int()

feaure_projecter = Project2DFeaturesCUDA(width=data_loader.dataset.depth_shape[1],
                            height=data_loader.dataset.depth_shape[0],
                            voxel_size = data_loader.dataset.VOXEL_SIZE,
                            config=config)


# Do projection
projected_feats, mapping2dto3d_num = feaure_projecter(encoded_images, coords, poses_2d, color_intrinsics)

for batch_id in coords[:, 0].unique():
    batch_mask = (coords[:, 0] == batch_id)  # * (mapping2dto3d_num > 0)

    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(coords[batch_mask, 1:].cpu().numpy())
    projected_pcd.colors = o3d.utility.Vector3dVector(projected_feats[batch_mask].cpu().numpy())

    orig_pcd = o3d.geometry.PointCloud()
    orig_pcd.points = o3d.utility.Vector3dVector(coords[batch_mask, 1:].cpu().numpy() + [1.5 * 50., 0., 0.])
    orig_pcd.colors = o3d.utility.Vector3dVector(colors[batch_mask].cpu().numpy())

    o3d.visualization.draw_geometries([projected_pcd + orig_pcd], window_name=f'Projected features for batch {batch_id}')