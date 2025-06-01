import torch
import numpy as np
from typing import Tuple, Optional, List
import open3d as o3d

class HashGrid:
    def __init__(self, 
                 cell_size: float,
                 hash_size: int = 2**20,
                 max_points_per_cell: int = 32):
        """
        Initialize a sparse hash grid for 3D points.
        
        Args:
            cell_size: Size of each grid cell
            hash_size: Size of the hash table (should be a power of 2)
            max_points_per_cell: Maximum number of points to store per cell
        """
        self.cell_size = cell_size
        self.hash_size = hash_size
        self.max_points_per_cell = max_points_per_cell
        
        # Hash table: maps cell coordinates to point indices
        self.hash_table = {}
        
        # Point storage
        self.points = None
        self.normals = None
        self.point_features = None
        
    def _get_cell_coords(self, points: torch.Tensor) -> torch.Tensor:
        """Convert 3D points to cell coordinates."""
        return torch.floor(points / self.cell_size).long()
    
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
             point_features: Optional[torch.Tensor] = None):
        """
        Build the hash grid from input points.
        
        Args:
            points: Input points (N, 3)
            normals: Point normals (N, 3)
            point_features: Additional point features (N, F)
        """
        self.points = points
        self.normals = normals
        self.point_features = point_features
        
        # Get cell coordinates for all points
        cell_coords = self._get_cell_coords(points)
        cell_hashes = self._hash_cell_coords(cell_coords)
        
        # Build hash table
        self.hash_table.clear()
        for i, cell_hash in enumerate(cell_hashes):
            cell_hash = cell_hash.item()
            if cell_hash not in self.hash_table:
                self.hash_table[cell_hash] = []
            if len(self.hash_table[cell_hash]) < self.max_points_per_cell:
                self.hash_table[cell_hash].append(i)
    
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
        if self.points is None:
            raise RuntimeError("Hash grid not built. Call build() first.")
        
        # Get cell coordinates for query points
        query_cell_coords = self._get_cell_coords(query_points)
        query_cell_hashes = self._hash_cell_coords(query_cell_coords)
        
        # For each query point, collect points from neighboring cells
        all_indices = []
        all_distances = []
        
        for i, (query_point, cell_hash) in enumerate(zip(query_points, query_cell_hashes)):
            # Get points from the same cell
            cell_hash = cell_hash.item()
            neighbor_indices = self.hash_table.get(cell_hash, [])
            
            # Get points from neighboring cells
            cell_coord = query_cell_coords[i]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        neighbor_cell = cell_coord + torch.tensor([dx, dy, dz], device=cell_coord.device)
                        neighbor_hash = self._hash_cell_coords(neighbor_cell.unsqueeze(0))[0].item()
                        neighbor_indices.extend(self.hash_table.get(neighbor_hash, []))
            
            # Remove duplicates
            neighbor_indices = list(set(neighbor_indices))
            
            if not neighbor_indices:
                all_indices.append(torch.zeros(k, dtype=torch.long, device=query_points.device))
                all_distances.append(torch.full((k,), float('inf'), device=query_points.device))
                continue
            
            # Compute distances
            neighbor_points = self.points[neighbor_indices]
            distances = torch.norm(neighbor_points - query_point, dim=1)
            
            # Apply radius filter if specified
            if radius is not None:
                mask = distances <= radius
                neighbor_indices = [idx for j, idx in enumerate(neighbor_indices) if mask[j]]
                distances = distances[mask]
            
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
            
            all_indices.append(torch.tensor(neighbor_indices, device=query_points.device))
            all_distances.append(distances)
        
        return torch.stack(all_indices), torch.stack(all_distances)
    
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
    
    def visualize_grid(self, save_path: str):
        """Visualize only the grid using Open3D."""
        if self.points is None:
            raise RuntimeError("Hash grid not built. Call build() first.")
        
        # Get grid bounds
        cell_coords = self._get_cell_coords(self.points)
        min_coords = cell_coords.min(dim=0)[0]
        max_coords = cell_coords.max(dim=0)[0]
        
        # Only visualize cells that contain points
        print("Collecting occupied cells...")
        occupied_cells = set()
        for i, cell_hash in enumerate(self._hash_cell_coords(cell_coords)):
            if cell_hash.item() in self.hash_table:
                occupied_cells.add(tuple(cell_coords[i].cpu().numpy()))
        
        print(f"Found {len(occupied_cells)} occupied cells")
        print("Creating grid visualization...")
        
        # Create grid lines
        grid_lines = []
        grid_points = []
        
        # Create grid lines only for occupied cells
        for cell_coord in occupied_cells:
            cell_coord = torch.tensor(cell_coord, device=self.points.device)
            # Add cell corners
            corners = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner = (cell_coord + torch.tensor([dx, dy, dz], device=self.points.device)) * self.cell_size
                        corners.append(corner.detach().cpu().numpy())
                        grid_points.append(corner.detach().cpu().numpy())
            
            # Add edges
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
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(grid_points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(grid_lines))
        
        # Set grid lines to red
        line_set.colors = o3d.utility.Vector3dVector(np.ones((len(grid_lines), 3)) * [1, 0, 0])
        
        # Save grid
        o3d.io.write_line_set(save_path, line_set)
        print(f"Saved grid visualization to {save_path}")
    
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
                            corner = (cell_coord + torch.tensor([dx, dy, dz], device=self.points.device)) * self.cell_size
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