import open3d as o3d
import numpy as np

# --- User config ---
PLY_IN = "combined_frustum_visualization.ply"
PLY_OUT = "combined_frustum_with_ray_line_ascii.ply"

# Ray points (world coordinates)
ray_points = np.array([
    (0.7962,0.9868,1.6248),
    (0.7784,1.0034,1.6346),
    (0.7606,1.0201,1.6445),
    (0.7428,1.0368,1.6543),
    (0.7249,1.0534,1.6641),
    (0.7071,1.0701,1.6740),
    (0.6893,1.0868,1.6838),
    (0.6715,1.1034,1.6936),
    (0.6537,1.1201,1.7035),
    (0.6358,1.1368,1.7133),
    (0.6180,1.1534,1.7231),
    (0.6002,1.1701,1.7329),
    (0.5824,1.1868,1.7428),
    (0.5646,1.2035,1.7526),
    (0.5467,1.2201,1.7624),
    (0.5289,1.2368,1.7723),
    (0.5111,1.2535,1.7821),
    (0.4933,1.2701,1.7919),
    (0.4755,1.2868,1.8018),
    (0.4576,1.3035,1.8116),
    (0.4398,1.3201,1.8214),
    (0.4220,1.3368,1.8313),
    (0.4042,1.3535,1.8411),
    (0.3864,1.3701,1.8509),
    (0.3685,1.3868,1.8608),
    (0.3507,1.4035,1.8706),
    (0.3329,1.4201,1.8804),
    (0.3151,1.4368,1.8903),
    (0.2973,1.4535,1.9001),
    (0.2794,1.4701,1.9099),
    (0.2616,1.4868,1.9197),
    (0.2438,1.5035,1.9296),
    (0.2260,1.5202,1.9394),
    (0.2081,1.5368,1.9492),
    (0.1903,1.5535,1.9591),
    (0.1725,1.5702,1.9689),
    (0.1547,1.5868,1.9787),
    (0.1369,1.6035,1.9886),
    (0.1190,1.6202,1.9984),
    (0.1012,1.6368,2.0082),
    (0.0834,1.6535,2.0181),
    (0.0656,1.6702,2.0279),
    (0.0478,1.6868,2.0377),
    (0.0299,1.7035,2.0476),
    (0.0121,1.7202,2.0574),
    (-0.0057,1.7368,2.0672),
    (-0.0235,1.7535,2.0770),
    (-0.0413,1.7702,2.0869),
    (-0.0592,1.7869,2.0967),
    (-0.0770,1.8035,2.1065),
    (-0.0948,1.8202,2.1164),
    (-0.1126,1.8369,2.1262),
    (-0.1304,1.8535,2.1360),
    (-0.1483,1.8702,2.1459),
    (-0.1661,1.8869,2.1557),
    (-0.1839,1.9035,2.1655),
    (-0.2017,1.9202,2.1754),
    (-0.2195,1.9369,2.1852),
    (-0.2374,1.9535,2.1950),
    (-0.2552,1.9702,2.2049),
    (-0.2730,1.9869,2.2147),
    (-0.2908,2.0035,2.2245),
    (-0.3086,2.0202,2.2344),
    (-0.3265,2.0369,2.2442),
    (-0.3443,2.0535,2.2540),
    (-0.3621,2.0702,2.2638),
    (-0.3799,2.0869,2.2737),
    (-0.3978,2.1036,2.2835),
    (-0.4156,2.1202,2.2933),
    (-0.4334,2.1369,2.3032),
    (-0.4512,2.1536,2.3130),
    (-0.4690,2.1702,2.3228),
    (-0.4869,2.1869,2.3327),
])


# Load the original PLY (binary)
p = o3d.io.read_point_cloud(PLY_IN)
orig_points = np.asarray(p.points)
orig_colors = np.asarray(p.colors)

# --- Color assignment ---
# Heuristic: If original points are not colored, set all to white
if orig_colors.shape[0] == 0 or np.allclose(orig_colors, 0):
    orig_colors = np.ones_like(orig_points)

# If original points are all the same color, set to white
if np.allclose(orig_colors, orig_colors[0]):
    orig_colors[:] = [1.0, 1.0, 1.0]

# Optionally, if you know the frustum points are a subset (e.g. first N or last N), you can set them to green here.
# For now, set all original points to white. If you want to color a subset green, adjust the indices below.

# Example: If you know the last 8 points are frustum corners, color them green
# orig_colors[-8:] = [0.0, 1.0, 0.0]

# Add ray points as red
ray_colors = np.tile([1.0, 0.0, 0.0], (ray_points.shape[0], 1))

# Combine points and colors
all_points = np.vstack([orig_points, ray_points])
all_colors = np.vstack([orig_colors, ray_colors])

# Create new point cloud
p_combined = o3d.geometry.PointCloud()
p_combined.points = o3d.utility.Vector3dVector(all_points)
p_combined.colors = o3d.utility.Vector3dVector(all_colors)

# Save as ASCII PLY
o3d.io.write_point_cloud(PLY_OUT, p_combined, write_ascii=True)

# Now write the polyline as a separate LineSet
lines = [[len(all_points)-ray_points.shape[0]+i, len(all_points)-ray_points.shape[0]+i+1] for i in range(ray_points.shape[0]-1)]
line_colors = [[1,0,0] for _ in lines]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(all_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(line_colors)

# Save the line set as a separate PLY (Open3D does not support writing lines in the same PLY as points)
o3d.io.write_line_set("ray_polyline_ascii.ply", line_set, write_ascii=True)

print("Wrote combined point cloud to {} (voxels white, ray red).".format(PLY_OUT))
print("If you want to color frustum points green, specify their indices in the script.")
