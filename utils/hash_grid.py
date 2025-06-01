import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
import open3d as o3d
import copy
from scipy.spatial import cKDTree

class HashGrid:
    def __init__(self, 
                 min_cell_size: float,
                 max_cell_size: float,
                 hash_size: int = 2**20,
                 max_points_per_cell: int = 32,
                 confidence_threshold: float = 0.5,
                 curvature_threshold: float = 0.1,
                 concentration_weight: float = 0.4,
                 density_weight: float = 0.3,
                 curvature_weight: float = 0.3):
        """
        Initialize a sparse hash grid for 3D points with adaptive cell sizing.
        
        Args:
            min_cell_size: Minimum cell size for dense/high-curvature regions
            max_cell_size: Maximum cell size for sparse/flat regions
            hash_size: Size of the hash table (should be a power of 2)
            max_points_per_cell: Maximum number of points to store per cell
            confidence_threshold: Minimum confidence score for points (0-1)
            curvature_threshold: Threshold for high curvature regions
            concentration_weight: Weight for spatial concentration in cell size computation
            density_weight: Weight for local density in cell size computation
            curvature_weight: Weight for curvature in cell size computation
        """
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.hash_size = hash_size
        self.max_points_per_cell = max_points_per_cell
        self.confidence_threshold = confidence_threshold
        self.curvature_threshold = curvature_threshold
        self.concentration_weight = concentration_weight
        self.density_weight = density_weight
        self.curvature_weight = curvature_weight
        
        # Hash table: maps cell coordinates to point indices
        self.hash_table = {}
        
        # Point storage
        self.points = None
        self.normals = None
        self.point_features = None
        self.confidence = None
        self.curvatures = None
        self.cell_sizes = None  # Store adaptive cell sizes for each point
        
    def compute_local_curvature(self, points: torch.Tensor, normals: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Compute local curvature using normal variation."""
        # Convert to numpy for KD-tree
        points_np = points.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()
        
        # Build KD-tree
        tree = cKDTree(points_np)
        
        # Find k-nearest neighbors for each point
        distances, indices = tree.query(points_np, k=k+1)  # +1 because point is its own neighbor
        
        # Compute curvature as normal variation
        curvatures = []
        for i in range(len(points)):
            neighbor_normals = normals_np[indices[i][1:]]  # Exclude self
            center_normal = normals_np[i]
            
            # Compute angle between center normal and neighbor normals
            angles = np.arccos(np.clip(np.dot(neighbor_normals, center_normal), -1.0, 1.0))
            curvature = np.mean(angles)
            curvatures.append(curvature)
            
        return torch.tensor(curvatures, device=points.device)
    
    def compute_local_density(self, points: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Compute local point density using k-nearest neighbors."""
        # Convert to numpy for KD-tree
        points_np = points.detach().cpu().numpy()
        
        # Build KD-tree
        tree = cKDTree(points_np)
        
        # Find k-nearest neighbors for each point
        distances, _ = tree.query(points_np, k=k+1)  # +1 because point is its own neighbor
        
        # Compute density as inverse of average distance to neighbors
        densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)  # Exclude self, add small epsilon
        densities = densities / np.max(densities)  # Normalize to [0, 1]
        
        return torch.tensor(densities, device=points.device)
    
    def compute_spatial_concentration(self, points: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Compute spatial concentration using k-nearest neighbors."""
        # Convert to numpy for KD-tree
        points_np = points.detach().cpu().numpy()
        
        # Build KD-tree
        tree = cKDTree(points_np)
        
        # Find k-nearest neighbors for each point
        distances, _ = tree.query(points_np, k=k+1)  # +1 because point is its own neighbor
        
        # Compute concentration as inverse of average distance to neighbors
        # Points in dense clusters will have smaller average distances
        concentration = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)  # Exclude self
        concentration = concentration / np.max(concentration)  # Normalize to [0, 1]
        
        return torch.tensor(concentration, device=points.device)
    
    def compute_adaptive_cell_sizes(self, 
                                  points: torch.Tensor, 
                                  normals: torch.Tensor,
                                  confidence: torch.Tensor) -> torch.Tensor:
        """Compute adaptive cell sizes based on concentration, density, and curvature."""
        # Compute spatial concentration
        concentration = self.compute_spatial_concentration(points)
        
        # Compute local density
        density = self.compute_local_density(points)
        
        # Compute local curvature
        curvature = self.compute_local_curvature(points, normals)
        
        # Normalize confidence to [0, 1] if not already
        if confidence.max() > 1.0:
            confidence = confidence / confidence.max()
        
        # Combine factors to compute cell size
        # - High concentration -> smaller cells
        # - High density -> smaller cells
        # - High curvature -> smaller cells
        concentration_factor = 1.0 - concentration
        density_factor = 1.0 - density
        curvature_factor = 1.0 - (curvature / curvature.max())
        
        # Weighted combination
        combined_factor = (self.concentration_weight * concentration_factor + 
                          self.density_weight * density_factor + 
                          self.curvature_weight * curvature_factor)
        
        # Map to cell size range
        cell_sizes = (self.max_cell_size * (1.0 - combined_factor) + 
                     self.min_cell_size * combined_factor)
        
        return cell_sizes
    
    def _get_cell_coords(self, points: torch.Tensor, cell_sizes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert 3D points to cell coordinates using adaptive cell sizes."""
        if cell_sizes is None:
            cell_sizes = torch.ones(len(points), device=points.device) * self.max_cell_size
        
        # Compute cell coordinates using adaptive cell sizes
        cell_coords = torch.floor(points / cell_sizes.unsqueeze(-1)).long()
        return cell_coords
    
    def _hash_cell_coords(self, cell_coords: torch.Tensor) -> torch.Tensor:
        """Hash cell coordinates to a 1D index."""
        # Use a simple hash function based on prime numbers
        p1 = 73856093
        p2 = 19349663
        p3 = 83492791
        
        x = cell_coords[:, 0]
        y = cell_coords[:, 1]
        z = cell_coords[:, 2]
        
        return ((x * p1) ^ (y * p2) ^ (z * p3)) % self.hash_size
    
    def build(self, 
             points: torch.Tensor,
             normals: Optional[torch.Tensor] = None,
             point_features: Optional[torch.Tensor] = None,
             confidence: Optional[torch.Tensor] = None):
        """
        Build the adaptive hash grid from input points.
        
        Args:
            points: Input points (N, 3)
            normals: Point normals (N, 3)
            point_features: Additional point features (N, F)
            confidence: Point confidence scores (N,) between 0 and 1
        """
        self.points = points
        self.normals = normals
        self.point_features = point_features
        
        # Set default confidence if not provided
        if confidence is None:
            confidence = torch.ones(len(points), device=points.device)
        self.confidence = confidence
        
        # Filter out low-confidence points
        mask = confidence > self.confidence_threshold
        filtered_points = points[mask]
        filtered_normals = normals[mask] if normals is not None else None
        filtered_confidence = confidence[mask]
        
        # Compute adaptive cell sizes
        self.cell_sizes = self.compute_adaptive_cell_sizes(
            filtered_points, 
            filtered_normals if filtered_normals is not None else torch.zeros_like(filtered_points),
            filtered_confidence
        )
        
        # Get cell coordinates using adaptive cell sizes
        cell_coords = self._get_cell_coords(filtered_points, self.cell_sizes)
        cell_hashes = self._hash_cell_coords(cell_coords)
        
        # Build hash table
        self.hash_table.clear()
        for i, cell_hash in enumerate(cell_hashes):
            cell_hash = cell_hash.item()
            if cell_hash not in self.hash_table:
                self.hash_table[cell_hash] = []
            if len(self.hash_table[cell_hash]) < self.max_points_per_cell:
                self.hash_table[cell_hash].append(i)
        
        # Store filtered points and attributes
        self.points = filtered_points
        self.normals = filtered_normals
        self.confidence = filtered_confidence
        
        # Print statistics
        print("\nAdaptive Hash Grid Statistics:")
        print(f"Total points: {len(points)}")
        print(f"Filtered points: {len(filtered_points)}")
        print(f"Number of cells: {len(self.hash_table)}")
        print(f"Average cell size: {self.cell_sizes.mean().item():.3f}")
        print(f"Min cell size: {self.cell_sizes.min().item():.3f}")
        print(f"Max cell size: {self.cell_sizes.max().item():.3f}")
    
    def query_points(self, 
                    query_points: torch.Tensor,
                    k: int = 8,
                    radius: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query k nearest neighbors for each query point.
        
        Args:
            query_points: Query points (M, 3)
            k: Number of neighbors to find
            radius: Optional maximum search radius
            
        Returns:
            Tuple of (indices, distances) for each query point
        """
        if self.points is None or len(self.points) == 0:
            raise RuntimeError("Hash grid not built or empty. Call build() first.")
        
        if len(query_points) == 0:
            return torch.zeros((0, k), dtype=torch.long, device=query_points.device), \
                   torch.zeros((0, k), dtype=torch.float, device=query_points.device)
        
        # Get cell coordinates for query points
        query_cell_coords = self._get_cell_coords(query_points)
        query_cell_hashes = self._hash_cell_coords(query_cell_coords)
        
        # Pre-allocate tensors for results
        all_indices = torch.full((len(query_points), k), -1, dtype=torch.long, device=query_points.device)
        all_distances = torch.full((len(query_points), k), float('inf'), dtype=torch.float, device=query_points.device)
        
        # Process queries in batches to manage memory
        batch_size = 100  # Adjust based on available memory
        for batch_start in range(0, len(query_points), batch_size):
            batch_end = min(batch_start + batch_size, len(query_points))
            batch_points = query_points[batch_start:batch_end]
            batch_cell_coords = query_cell_coords[batch_start:batch_end]
            batch_cell_hashes = query_cell_hashes[batch_start:batch_end]
            
            for i, (query_point, cell_hash) in enumerate(zip(batch_points, batch_cell_hashes)):
                try:
                    # Get points from the same cell
                    cell_hash = cell_hash.item()
                    neighbor_indices = set(self.hash_table.get(cell_hash, []))
                    
                    # Get points from neighboring cells (only if needed)
                    if len(neighbor_indices) < k:
                        cell_coord = batch_cell_coords[i]
                        # Check neighboring cells in order of increasing distance
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == dy == dz == 0:
                                        continue
                                    neighbor_cell = cell_coord + torch.tensor([dx, dy, dz], device=cell_coord.device)
                                    neighbor_hash = self._hash_cell_coords(neighbor_cell.unsqueeze(0))[0].item()
                                    neighbor_indices.update(self.hash_table.get(neighbor_hash, []))
                    
                    if not neighbor_indices:
                        continue
                    
                    # Convert to list and compute distances
                    neighbor_indices = list(neighbor_indices)
                    neighbor_points = self.points[neighbor_indices]
                    distances = torch.norm(neighbor_points - query_point, dim=1)
                    
                    # Apply radius filter if specified
                    if radius is not None:
                        mask = distances <= radius
                        neighbor_indices = [idx for j, idx in enumerate(neighbor_indices) if mask[j]]
                        distances = distances[mask]
                    
                    if not neighbor_indices:
                        continue
                    
                    # Get k nearest neighbors
                    if len(neighbor_indices) > k:
                        k_indices = torch.topk(distances, k, largest=False)[1]
                        neighbor_indices = [neighbor_indices[idx] for idx in k_indices]
                        distances = distances[k_indices]
                    else:
                        # Pad with -1 if not enough neighbors
                        pad_size = k - len(neighbor_indices)
                        neighbor_indices.extend([-1] * pad_size)
                        distances = torch.cat([distances, torch.full((pad_size,), float('inf'), device=distances.device)])
                    
                    # Store results
                    all_indices[batch_start + i] = torch.tensor(neighbor_indices, device=query_points.device)
                    all_distances[batch_start + i] = distances
                    
                except Exception as e:
                    print(f"Warning: Error processing query point {batch_start + i}: {str(e)}")
                    continue
        
        return all_indices, all_distances
    
    def get_cell_points(self, cell_coord: torch.Tensor) -> List[int]:
        """Get all point indices in a specific cell."""
        cell_hash = self._hash_cell_coords(cell_coord.unsqueeze(0))[0].item()
        return self.hash_table.get(cell_hash, [])
    
    def visualize_points(self, save_path: str):
        """Visualize only the points using Open3D."""
        if self.points is None:
            raise RuntimeError("Hash grid not built. Call build() first.")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.detach().cpu().numpy())
        
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals.detach().cpu().numpy())
        
        # Set points to white
        pcd.colors = o3d.utility.Vector3dVector(np.ones((len(self.points), 3)) * [1, 1, 1])
        
        # Save point cloud
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Saved point cloud to {save_path}")
    
    def visualize_grid(self, save_path=None):
        """Visualize the hash grid as wireframe cubes for non-empty voxels."""
        if self.points is None:
            raise RuntimeError("Hash grid not built. Call build() first.")
            
        # Get cell coordinates for all points
        cell_coords = self._get_cell_coords(self.points, self.cell_sizes)
        cell_hashes = self._hash_cell_coords(cell_coords)
        
        # Collect occupied cells and their average cell sizes and point counts
        occupied_cells = {}  # Map cell coordinates to (avg_cell_size, point_count)
        for i, (cell_coord, cell_hash) in enumerate(zip(cell_coords, cell_hashes)):
            if cell_hash.item() in self.hash_table:
                cell_key = tuple(cell_coord.cpu().numpy())
                if cell_key not in occupied_cells:
                    occupied_cells[cell_key] = {"sizes": [], "count": 0}
                occupied_cells[cell_key]["sizes"].append(self.cell_sizes[i].item())
                occupied_cells[cell_key]["count"] += 1
        
        # Compute statistics
        point_counts = [cell_info["count"] for cell_info in occupied_cells.values()]
        min_points = min(point_counts)
        max_points = max(point_counts)
        avg_points = sum(point_counts) / len(point_counts)
        
        # Compute average cell size for each occupied cell
        for cell_key in occupied_cells:
            occupied_cells[cell_key]["avg_size"] = sum(occupied_cells[cell_key]["sizes"]) / len(occupied_cells[cell_key]["sizes"])
        
        num_voxels = len(occupied_cells)
        print("\nVoxel Grid Statistics:")
        print(f"Total number of voxels: {num_voxels}")
        print(f"Average cell size: {sum(cell['avg_size'] for cell in occupied_cells.values()) / num_voxels:.3f}")
        print(f"Total number of points: {len(self.points)}")
        print(f"Points per voxel:")
        print(f"  Minimum: {min_points}")
        print(f"  Maximum: {max_points}")
        print(f"  Average: {avg_points:.1f}")
        
        # Create wireframe visualization with thinner lines
        vertices = []
        triangles = []
        vertex_colors = []
        
        # Create wireframe for each occupied cell
        for cell_coord, cell_info in occupied_cells.items():
            cell_coord = torch.tensor(cell_coord, device=self.points.device)
            cell_size = cell_info["avg_size"]
            
            # Get cell corners
            corners = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner = (cell_coord + torch.tensor([dx, dy, dz], device=self.points.device)) * cell_size
                        corners.append(corner.detach().cpu().numpy())
            
            # Define edges of the cube (12 edges)
            edges = [
                (0, 1), (0, 2), (0, 4),  # Front face
                (1, 3), (1, 5),          # Right face
                (2, 3), (2, 6),          # Back face
                (3, 7),                  # Top face
                (4, 5), (4, 6),          # Bottom face
                (5, 7), (6, 7)           # Left face
            ]
            
            # For each edge, create a thin cylinder mesh
            # Use thinner lines (reduced radius)
            for edge in edges:
                start = corners[edge[0]]
                end = corners[edge[1]]
                # Create a thinner cylinder for the edge (reduced radius from 0.02 to 0.005)
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=1.0)
                # Compute rotation to align cylinder with edge
                direction = end - start
                length = np.linalg.norm(direction)
                if length > 0:
                    # Normalize direction
                    direction = direction / length
                    # Create rotation matrix
                    z_axis = np.array([0, 0, 1])
                    rotation_axis = np.cross(z_axis, direction)
                    if np.linalg.norm(rotation_axis) > 0:
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        rotation_angle = np.arccos(np.dot(z_axis, direction))
                        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                        cylinder.rotate(rotation_matrix, center=[0, 0, 0])
                    # Scale and translate
                    cylinder.scale(length, center=[0, 0, 0])
                    cylinder.translate((start + end) / 2)
                    # Add to mesh
                    vertices.extend(np.asarray(cylinder.vertices))
                    triangles.extend(np.asarray(cylinder.triangles) + len(vertices) - len(cylinder.vertices))
                    # Use a more subtle red color
                    vertex_colors.extend(np.ones((len(cylinder.vertices), 3)) * [0.8, 0.2, 0.2])
        
        if len(vertices) == 0:
            print("No voxels to visualize!")
            return
            
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
        
        if save_path is not None:
            print(f"Saving grid visualization to {save_path}")
            o3d.io.write_triangle_mesh(save_path, mesh)
        else:
            # Visualize
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)
            
            # Set up camera
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            # Run visualizer
            vis.run()
            vis.destroy_window()
    
    def visualize(self, save_path: Optional[str] = None):
        """Visualize both points and grid using Open3D."""
        if save_path is not None:
            self.visualize_points(save_path + "_points.ply")
            self.visualize_grid(save_path + "_grid.ply")
        else:
            # For interactive visualization, we need both
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points.detach().cpu().numpy())
            if self.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(self.normals.detach().cpu().numpy())
            
            # Set points to white
            pcd.colors = o3d.utility.Vector3dVector(np.ones((len(self.points), 3)) * [1, 1, 1])
            
            # Get grid visualization
            cell_coords = self._get_cell_coords(self.points)
            occupied_cells = set()
            for i, cell_hash in enumerate(self._hash_cell_coords(cell_coords)):
                if cell_hash.item() in self.hash_table:
                    occupied_cells.add(tuple(cell_coords[i].cpu().numpy()))
            
            grid_lines = []
            grid_points = []
            
            for cell_coord in occupied_cells:
                cell_coord = torch.tensor(cell_coord, device=self.points.device)
                corners = []
                for dx in [0, 1]:
                    for dy in [0, 1]:
                        for dz in [0, 1]:
                            corner = (cell_coord + torch.tensor([dx, dy, dz], device=self.points.device)) * self.cell_sizes[cell_coord[0], cell_coord[1], cell_coord[2]].item()
                            corners.append(corner.detach().cpu().numpy())
                            grid_points.append(corner.detach().cpu().numpy())
                
                edges = [
                    (0, 1), (0, 2), (0, 4),
                    (1, 3), (1, 5),
                    (2, 3), (2, 6),
                    (3, 7),
                    (4, 5), (4, 6),
                    (5, 7),
                    (6, 7)
                ]
                
                for edge in edges:
                    grid_lines.append([len(grid_points) - 8 + edge[0], len(grid_points) - 8 + edge[1]])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(grid_points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(grid_lines))
            
            # Set grid lines to red
            line_set.colors = o3d.utility.Vector3dVector(np.ones((len(grid_lines), 3)) * [1, 0, 0])
            
            o3d.visualization.draw_geometries([pcd, line_set]) 