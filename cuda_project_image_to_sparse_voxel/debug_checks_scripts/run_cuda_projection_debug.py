import torch
import numpy as np
import open3d as o3d
import os

def create_camera_frustum_lines(K, pose, img_w, img_h, scale=1.0, color=[0, 0, 1]):
    fx, fy, mx, my = K
    corners = np.array([
        [0, 0], [img_w-1, 0], [img_w-1, img_h-1], [0, img_h-1]
    ])
    cam_pts = np.stack([
        (corners[:,0] - mx) / fx,
        (corners[:,1] - my) / fy,
        np.ones(4)
    ], axis=1) * scale
    cam_pts = cam_pts.T
    cam_pts = np.concatenate([np.zeros((3,1)), cam_pts], axis=1)
    cam_pts_h = np.concatenate([cam_pts, np.ones((1,5))], axis=0)
    world_pts = (pose @ cam_pts_h)[:3].T
    lines = [
        [0,1],[0,2],[0,3],[0,4],
        [1,2],[2,3],[3,4],[4,1]
    ]
    colors = [color for _ in lines]
    return world_pts, lines, colors

def create_ray_line(origin, direction, t_near, t_far, color=[1,0,0]):
    p0 = origin + t_near * direction
    p1 = origin + t_far * direction
    return np.stack([p0, p1]), [[0,1]], [color]

def debug_tensor(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    print(f"  min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.float().mean().item()}")
    print(f"  sample values: {tensor.flatten()[:10].cpu().numpy()}")

def save_point_cloud(filename, points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud: {filename}")

def save_line_set(filename, points, lines, colors):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(filename, line_set)
    print(f"Saved line set: {filename}")

def main():
    encoded_2d_features = torch.load('input/encoded_2d_features.pt')
    occupancy_3D = torch.load('input/occupancy_3D.pt')
    viewMatrixInv = torch.load('input/viewMatrixInv.pt')
    intrinsicParams = torch.load('input/intrinsicParams.pt')
    opts = torch.tensor([encoded_2d_features.shape[3], encoded_2d_features.shape[2], 0.1, 10.0, 0.05], dtype=torch.float32)
    pred_mode_t = torch.tensor([False], dtype=torch.bool)

    debug_tensor("encoded_2d_features", encoded_2d_features)
    debug_tensor("occupancy_3D", occupancy_3D)
    debug_tensor("viewMatrixInv", viewMatrixInv)
    debug_tensor("intrinsicParams", intrinsicParams)
    print(f"opts: {opts}")
    print(f"pred_mode_t: {pred_mode_t}")

    B, V, H, W, C = encoded_2d_features.shape
    Z, Y, X = occupancy_3D.shape[1:]
    num_voxels = B * Z * Y * X
    mapping2dto3d_num = torch.zeros(num_voxels, dtype=torch.int32, device='cuda')
    projected_features = torch.zeros(num_voxels, C, dtype=torch.float32, device='cuda')

    encoded_2d_features = encoded_2d_features.cuda()
    occupancy_3D = occupancy_3D.cuda()
    viewMatrixInv = viewMatrixInv.cuda()
    intrinsicParams = intrinsicParams.cuda()
    opts = opts.cuda()
    pred_mode_t = pred_mode_t.cuda()

    import project_features_cuda as cuda_proj
    cuda_proj.project_features_cuda_forward_impl(
        encoded_2d_features,
        occupancy_3D,
        viewMatrixInv,
        intrinsicParams,
        opts,
        mapping2dto3d_num,
        projected_features,
        pred_mode_t
    )
    print("CUDA projection done.")

    debug_tensor("projected_features", projected_features)
    debug_tensor("mapping2dto3d_num", mapping2dto3d_num)

    occupancy_mask = (occupancy_3D.flatten() > 0).cpu().numpy()
    voxel_indices = np.argwhere(occupancy_mask)
    coords = []
    for idx in voxel_indices:
        b = idx // (Z*Y*X)
        rem = idx % (Z*Y*X)
        z = rem // (Y*X)
        y = (rem % (Y*X)) // X
        x = rem % X
        coords.append([x, y, z])
    coords = np.array(coords)
    save_point_cloud("output/occupied_voxels.ply", coords)

    features = projected_features.cpu().numpy()
    if features.shape[1] >= 3:
        colors = features[occupancy_mask, :3]
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
    else:
        c = features[occupancy_mask, 0]
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        colors = np.stack([c, c, c], axis=1)
    save_point_cloud("output/projected_features_colored.ply", coords, colors)

    os.makedirs("output", exist_ok=True)
    for b in range(B):
        for v in range(V):
            pose = viewMatrixInv[b, v].cpu().numpy()
            K = intrinsicParams[b].cpu().numpy()
            frustum_pts, frustum_lines, frustum_colors = create_camera_frustum_lines(
                K, pose, W, H, scale=opts[2].item(), color=[0,0,1]
            )
            save_line_set(f"output/frustum_b{b}_v{v}.ply", frustum_pts, frustum_lines, frustum_colors)
            ray_lines_pts = []
            ray_lines = []
            ray_colors = []
            ray_id = 0
            for yi in np.linspace(0, H-1, 5):
                for xi in np.linspace(0, W-1, 5):
                    fx, fy, mx, my = K
                    x_cam = (xi - mx) / fx
                    y_cam = (yi - my) / fy
                    z_cam = 1.0
                    cam_dir = np.array([x_cam, y_cam, z_cam])
                    cam_dir = cam_dir / np.linalg.norm(cam_dir)
                    cam_dir_h = np.concatenate([cam_dir, [0]])
                    world_dir = (pose @ cam_dir_h)[:3]
                    world_dir = world_dir / np.linalg.norm(world_dir)
                    cam_origin = (pose @ np.array([0,0,0,1]))[:3]
                    pts, lines, colors = create_ray_line(cam_origin, world_dir, opts[2].item(), opts[3].item(), color=[1,0,0])
                    pts_idx = np.arange(ray_id, ray_id+2)
                    ray_lines_pts.append(pts)
                    ray_lines.extend([[pts_idx[0], pts_idx[1]]])
                    ray_colors.extend(colors)
                    ray_id += 2
            if ray_lines_pts:
                ray_lines_pts = np.concatenate(ray_lines_pts, axis=0)
                save_line_set(f"output/rays_b{b}_v{v}.ply", ray_lines_pts, ray_lines, ray_colors)

if __name__ == "__main__":
    main()
