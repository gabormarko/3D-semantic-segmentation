import torch
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Optional
import open3d as o3d

class SurfaceDetector:
    def __init__(self, 
                 opacity_threshold: float = 0.5,
                 scale_threshold: float = 0.1,
                 density_threshold: float = 0.1,
                 k_neighbors: int = 8):
        """
        Initialize the surface detector.
        
        Args:
            opacity_threshold: Minimum opacity for a Gaussian to be considered
            scale_threshold: Maximum scale for a Gaussian to be considered (relative to scene extent)
            density_threshold: Minimum local density for a point to be considered part of the surface
            k_neighbors: Number of neighbors to consider for density estimation
        """
        self.opacity_threshold = opacity_threshold
        self.scale_threshold = scale_threshold
        self.density_threshold = density_threshold
        self.k_neighbors = k_neighbors

    def filter_gaussians(self, 
                        xyz: torch.Tensor,
                        opacity: torch.Tensor,
                        scaling: torch.Tensor,
                        scene_extent: float) -> torch.Tensor:
        """
        Filter Gaussians based on opacity and scale.
        
        Args:
            xyz: Gaussian positions (N, 3)
            opacity: Gaussian opacities (N, 1)
            scaling: Gaussian scales (N, 3)
            scene_extent: Scene extent for scale normalization
            
        Returns:
            Boolean mask of valid Gaussians
        """
        # Filter by opacity
        opacity_mask = opacity.squeeze() > self.opacity_threshold
        
        # Filter by scale (relative to scene extent)
        max_scales = torch.max(scaling, dim=1).values
        scale_mask = max_scales < (self.scale_threshold * scene_extent)
        
        # Combine masks
        valid_mask = torch.logical_and(opacity_mask, scale_mask)
        
        return valid_mask

    def compute_local_density(self, 
                            xyz: torch.Tensor,
                            valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute local density for each point using k-nearest neighbors.
        
        Args:
            xyz: Gaussian positions (N, 3)
            valid_mask: Boolean mask of valid Gaussians
            
        Returns:
            Local density values for each point
        """
        # Convert to numpy for KDTree
        valid_xyz = xyz[valid_mask].detach().cpu().numpy()
        
        # Build KDTree
        tree = KDTree(valid_xyz)
        
        # Query k-nearest neighbors
        distances, _ = tree.query(valid_xyz, k=self.k_neighbors + 1)  # +1 because point is its own neighbor
        
        # Compute local density as inverse of mean distance to k-nearest neighbors
        # Exclude self-distance (first column)
        local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
        
        # Convert back to tensor
        density = torch.zeros(xyz.shape[0], device=xyz.device)
        density[valid_mask] = torch.from_numpy(local_density.astype(np.float32)).to(xyz.device)
        
        return density

    def extract_surface_points(self,
                             xyz: torch.Tensor,
                             opacity: torch.Tensor,
                             scaling: torch.Tensor,
                             scene_extent: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract surface points from Gaussians.
        
        Args:
            xyz: Gaussian positions (N, 3)
            opacity: Gaussian opacities (N, 1)
            scaling: Gaussian scales (N, 3)
            scene_extent: Scene extent for scale normalization
            
        Returns:
            Tuple of (surface_points, surface_normals)
        """
        # Filter Gaussians
        valid_mask = self.filter_gaussians(xyz, opacity, scaling, scene_extent)
        
        # Compute local density
        density = self.compute_local_density(xyz, valid_mask)
        
        # Filter by density
        density_mask = density > self.density_threshold
        surface_mask = torch.logical_and(valid_mask, density_mask)
        
        # Get surface points
        surface_points = xyz[surface_mask]
        
        # Estimate normals using PCA on local neighborhood
        surface_normals = self.estimate_normals(surface_points)
        
        return surface_points, surface_normals

    def estimate_normals(self, points: torch.Tensor) -> torch.Tensor:
        """
        Estimate surface normals using PCA on local neighborhood.
        
        Args:
            points: Surface points (N, 3)
            
        Returns:
            Estimated normals (N, 3)
        """
        # Convert to numpy for Open3D
        points_np = points.detach().cpu().numpy()
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.scale_threshold * 2.0,
                max_nn=self.k_neighbors
            )
        )
        
        # Get normals
        normals = torch.from_numpy(np.asarray(pcd.normals)).to(points.device)
        
        return normals

    def visualize_surface(self,
                         surface_points: torch.Tensor,
                         surface_normals: Optional[torch.Tensor] = None,
                         save_path: Optional[str] = None):
        """
        Visualize the extracted surface points and normals.
        
        Args:
            surface_points: Surface points (N, 3)
            surface_normals: Surface normals (N, 3)
            save_path: Optional path to save visualization
        """
        # Convert to numpy
        points_np = surface_points.detach().cpu().numpy()
        normals_np = surface_normals.detach().cpu().numpy() if surface_normals is not None else None
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        
        if normals_np is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals_np)
        
        # print("Point cloud has", len(points_np), "points.")
        if save_path is not None:
            o3d.io.write_point_cloud(save_path, pcd)
        else:
            print("Visualizing surface (Open3D window) – press q to exit.")
            o3d.visualization.draw_geometries([pcd], window_name="Surface Visualization (press q to exit)") 