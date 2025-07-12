


# --- Combine ray polyline with existing PLY ---
PLY_OUT = "combined_frustum_with_ray_line.ply"
EXISTING_PLY = "combined_frustum_visualization.ply"
ray_points = [
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
]

def combine_ply_with_ray_line(existing_ply, ray_points, out_ply):
    # Read existing ply
    with open(existing_ply, 'r', encoding='ascii', errors='ignore') as f:
        ply_lines = f.readlines()
    # Find header and vertex section
    end_header_idx = [i for i, l in enumerate(ply_lines) if l.strip() == 'end_header'][0]
    vertex_count_line_idx = [i for i, l in enumerate(ply_lines) if l.startswith('element vertex')][0]
    vertex_count = int(ply_lines[vertex_count_line_idx].split()[-1])
    # Check for edge element
    edge_count_line_idx = next((i for i, l in enumerate(ply_lines) if l.startswith('element edge')), None)
    edge_count = 0
    if edge_count_line_idx is not None:
        edge_count = int(ply_lines[edge_count_line_idx].split()[-1])
    # The vertex section is immediately after the header
    vertex_start_idx = end_header_idx + 1
    vertex_end_idx = vertex_start_idx + vertex_count
    # Add ray points as red
    ray_vertex_lines = [f"{x:.6f} {y:.6f} {z:.6f} 255 0 0\n" for x, y, z in ray_points]
    # Edges: connect all new points in a polyline (as consecutive edges)
    n_points = len(ray_points)
    ray_edge_lines = [f"{vertex_count + i} {vertex_count + i + 1} 255 0 0\n" for i in range(n_points-1)]
    # Update header
    new_vertex_count = vertex_count + n_points
    new_edge_count = edge_count + (n_points - 1)
    ply_lines[vertex_count_line_idx] = f'element vertex {new_vertex_count}\n'
    if edge_count_line_idx is not None:
        ply_lines[edge_count_line_idx] = f'element edge {new_edge_count}\n'
    else:
        # Insert edge element before end_header
        ply_lines.insert(end_header_idx, f'element edge {n_points-1}\n')
        ply_lines.insert(end_header_idx+1, 'property int vertex1\n')
        ply_lines.insert(end_header_idx+2, 'property int vertex2\n')
        ply_lines.insert(end_header_idx+3, 'property uchar red\n')
        ply_lines.insert(end_header_idx+4, 'property uchar green\n')
        ply_lines.insert(end_header_idx+5, 'property uchar blue\n')
        end_header_idx += 6
        vertex_start_idx += 6
        vertex_end_idx += 6
    # Insert ray points after all existing vertices
    ply_lines = ply_lines[:vertex_end_idx] + ray_vertex_lines + ply_lines[vertex_end_idx:]
    # Find where to insert edges (after all vertices)
    # Find the end of the file or where existing edges start
    if edge_count_line_idx is not None:
        # Edges start after all vertices
        edge_start_idx = vertex_end_idx + n_points
        ply_lines = ply_lines[:edge_start_idx] + ray_edge_lines + ply_lines[edge_start_idx:]
    else:
        # No existing edges, just append at the end
        ply_lines += ray_edge_lines
    # Write out
    with open(out_ply, 'w') as f:
        f.writelines(ply_lines)

combine_ply_with_ray_line(EXISTING_PLY, ray_points, PLY_OUT)
print(f"Wrote combined PLY with ray line to {PLY_OUT}")
