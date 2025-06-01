import torch
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Optional
import open3d as o3d
from sklearn.cluster import DBSCAN

class SurfaceDetector:
    def __init__(self, 
                 opacity_threshold: float = 0.8,
                 scale_threshold: float = 0.1,
                 density_threshold: float = 0.1,
                 k_neighbors: int = 16,
                 spatial_concentration_threshold: float = 0.3,
                 min_cluster_size: int = 100):
        """
        Initialize the surface detector.
        
        Args:
            opacity_threshold: Minimum opacity for a Gaussian to be considered
            scale_threshold: Maximum scale for a Gaussian to be considered (relative to scene extent)
            density_threshold: Minimum local density for a point to be considered part of the surface
            k_neighbors: Number of neighbors to consider for density estimation
            spatial_concentration_threshold: Threshold for spatial concentration (0-1)
            min_cluster_size: Minimum number of points in a cluster to be considered valid
        """
        self.opacity_threshold = opacity_threshold
        self.scale_threshold = scale_threshold
        self.density_threshold = density_threshold
        self.k_neighbors = k_neighbors
        self.spatial_concentration_threshold = spatial_concentration_threshold
        self.min_cluster_size = min_cluster_size

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

    def compute_spatial_concentration(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial concentration score for each point.
        Points in dense clusters get higher scores, isolated points get lower scores.
        """
        # Convert to numpy for DBSCAN
        points_np = xyz.detach().cpu().numpy()
        
        # Compute pairwise distances
        clustering = DBSCAN(eps=self.scale_threshold * 2, min_samples=5).fit(points_np)
        labels = clustering.labels_
        
        # Count points in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        # Assign concentration scores based on cluster size
        concentration = torch.zeros(len(points_np), device=xyz.device)
        for i, label in enumerate(labels):
            if label != -1:  # Not noise
                size = cluster_sizes[label]
                concentration[i] = min(1.0, size / self.min_cluster_size)
        
        return concentration

    def filter_background_points(self, 
                               xyz: torch.Tensor,
                               opacity: torch.Tensor,
                               scaling: torch.Tensor,
                               scene_extent: float) -> torch.Tensor:
        """
        Filter out background points by analyzing spatial distribution and density.
        
        Args:
            xyz: Gaussian positions (N, 3)
            opacity: Gaussian opacities (N, 1)
            scaling: Gaussian scales (N, 3)
            scene_extent: Scene extent for scale normalization
            
        Returns:
            Boolean mask of points to keep (True for foreground points)
        """
        # Basic filtering by opacity and scale
        valid_mask = self.filter_gaussians(xyz, opacity, scaling, scene_extent)
        
        # Compute local density
        density = self.compute_local_density(xyz, valid_mask)
        
        # Compute spatial concentration
        concentration = self.compute_spatial_concentration(xyz)
        
        # Combine masks
        density_mask = density > self.density_threshold
        concentration_mask = concentration > self.spatial_concentration_threshold
        
        # Points must satisfy all criteria
        foreground_mask = torch.logical_and(valid_mask, density_mask)
        foreground_mask = torch.logical_and(foreground_mask, concentration_mask)
        
        return foreground_mask

    def extract_surface_points(self,
                             xyz: torch.Tensor,
                             opacity: torch.Tensor,
                             scaling: torch.Tensor,
                             scene_extent: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract surface points from Gaussians, focusing on concentrated geometry.
        
        Args:
            xyz: Gaussian positions (N, 3)
            opacity: Gaussian opacities (N, 1)
            scaling: Gaussian scales (N, 3)
            scene_extent: Scene extent for scale normalization
            
        Returns:
            Tuple of (surface_points, surface_normals)
        """
        # Filter out background points
        foreground_mask = self.filter_background_points(xyz, opacity, scaling, scene_extent)
        
        # Get foreground points
        foreground_points = xyz[foreground_mask]
        
        # Compute local density for remaining points
        density = self.compute_local_density(foreground_points, torch.ones(len(foreground_points), dtype=torch.bool, device=xyz.device))
        
        # Filter by density
        density_mask = density > self.density_threshold
        surface_mask = density_mask
        
        # Get surface points
        surface_points = foreground_points[surface_mask]
        
        # Estimate normals using PCA on local neighborhood
        surface_normals = self.estimate_normals(surface_points)
        
        print(f"\nSurface Detection Statistics:")
        print(f"Total points: {len(xyz)}")
        print(f"Foreground points: {len(foreground_points)}")
        print(f"Surface points: {len(surface_points)}")
        
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