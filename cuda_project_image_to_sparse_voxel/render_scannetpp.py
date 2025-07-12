#!/usr/bin/env python3
"""
Render per‑image semantic masks for a ScanNet++ scene so they can be used in the
Unified‑Lift pipeline.

The script does four main things:
 1. Loads the semantic mesh that contains a per‑vertex `label` property
    (e.g. scans/mesh_aligned_0.05_semantic.ply).
 2. Builds a *dense* scene‑specific mapping from the global ScanNet++ IDs to
    0 … K‑1 (and stores it in label_mapping.json).  The ignore label –100 is
    mapped to 255.
 3. Uses COLMAP intrinsics / extrinsics from sparse/0/{cameras,images}.bin to
    render a 2‑D label image that is pixel‑aligned with each RGB frame under
    images/.
 4. Saves the masks as uint8 PNGs in semantic_mask/ with the same file names
    as the RGB images.

Dependencies (install once):
    pip install numpy open3d plyfile imageio

Usage:
    python render_semantic_masks.py --scene_dir /path/to/scene_X

If your Open3D is < 0.17, the OffscreenRenderer API may differ slightly; in
that case follow the comments marked "*****" for alternate code.
"""

#!/usr/bin/env python3
import json, struct, argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import open3d as o3d
from plyfile import PlyData


# -------------------------------------------------------------------
# ----------  COLMAP binary helpers  --------------------------------
def read_cameras_binary(path):
    MODEL_NUM_PARAMS = {0:3, 1:4, 2:3, 3:4, 4:8, 5:8, 6:12, 7:5, 8:5, 9:8, 10:4}
    cams = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("Q", f.read(8))[0]):
            cid  = struct.unpack("I", f.read(4))[0]
            mid  = struct.unpack("i", f.read(4))[0]
            w,h  = struct.unpack("QQ", f.read(16))
            k    = MODEL_NUM_PARAMS[mid]
            pars = struct.unpack(f"{k}d", f.read(8*k))
            cams[cid] = {"model_id":mid, "width":w, "height":h,
                         "params":np.array(pars,dtype=np.float64)}
    return cams



def read_images_binary(path):
    imgs = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("Q", f.read(8))[0]):
            iid  = struct.unpack("I", f.read(4))[0]
            qvec = struct.unpack("dddd", f.read(32))
            tvec = struct.unpack("ddd",  f.read(24))
            cid  = struct.unpack("I",   f.read(4))[0]
            name = b""
            while (b:=f.read(1)) != b"\x00": name += b
            n2d  = struct.unpack("Q", f.read(8))[0]
            f.read(24*n2d)                          # skip points
            imgs[iid] = {"qvec":np.array(qvec), "tvec":np.array(tvec),
                         "cam_id":cid, "name":name.decode()}
    return imgs


def qvec_to_R(q):
    qw,qx,qy,qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]])


# -------------------------------------------------------------------
def build_dense_map(raw, void=-100, ignore=255):
    uniq = np.unique(raw)
    uniq = uniq[uniq!=void]
    m = {int(lbl):i for i,lbl in enumerate(uniq)}
    m[void] = ignore
    return m


def remap(arr,m,ignore=255):
    out = np.full_like(arr, ignore, dtype=np.uint16)
    for k,v in m.items(): out[arr==k]=v
    return out

def make_random_blue_channel(dense: np.ndarray,
                             seed: int = 123,
                             ignore_label: int = 255) -> np.ndarray:
    """
    Parameters
    ----------
    dense : np.ndarray
        The uint16/uint8 array of dense labels after `remap(...)`.
    seed : int
        Seed for reproducibility.  Change it if you want a different colour set.
    ignore_label : int
        Label that should stay at blue=0 (background / void).

    Returns
    -------
    blue : np.ndarray  (same shape as `dense`, dtype uint8)
        Per-vertex (or per-pixel) blue-channel values in 0‒255.
    """
    rng = np.random.default_rng(seed)

    # full LUT, one entry per possible label value
    lut = rng.integers(0, 256, size=256, dtype=np.uint8)
    lut[ignore_label] = 0          # keep “ignore / void” dark

    # map every label through the LUT in one shot
    return lut[dense.astype(np.uint8)]
# -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def render_scene_masks(scene_dir: Path, out_dir: Path):
    scans_dir  = scene_dir / "scans"
    sparse_dir = scene_dir / "sparse/0"
    img_dir    = scene_dir / "images"                 # (not used but kept)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load mesh & labels ----------
    print("Loading semantic mesh …")
    ply  = PlyData.read(str(scans_dir / "mesh_aligned_0.05_semantic.ply"))
    V    = np.stack([ply["vertex"][ax] for ax in ("x","y","z")], axis=-1).astype(np.float32)
    F    = np.stack(ply["face"]["vertex_indices"], axis=0).astype(np.int32)
    raw  = np.array(ply["vertex"]["label"], dtype=np.int32)
    # ---------- dense label map ----------
    mapping = build_dense_map(raw)
    (scene_dir / "label_mapping.json").write_text(json.dumps(mapping,indent=2))
    dense   = remap(raw, mapping)          # uint16 but ≤ 255
    assert dense.max() < 256, "labels already checked earlier"

    # ---------- encode as vertex colour (uint8) ----------
    # ---------- build a FLAT-SHADED mesh (one colour per triangle) ----------
    faces  = np.asarray(F)                       # (N_faces, 3) indices
    labels = dense[faces]                        # (N_faces, 3) vertex labels

    # majority vote per triangle  (ties -> smallest label)
    maj_lbl = np.array([np.bincount(t).argmax() for t in labels], dtype=np.uint8)

    # deterministic pseudo-random RGB for each semantic label
    def label_to_rgb(lbl, seed=42):
        rng = np.random.default_rng(lbl + seed)
        return rng.integers(0, 256, 3, dtype=np.uint8)

    tri_col = np.array([label_to_rgb(l) for l in maj_lbl], dtype=np.uint8)  # (N_faces,3)

    # duplicate verts so each triangle is isolated (no colour blending)
    flat_V = V[faces].reshape(-1, 3)              # (N_faces*3, 3)
    flat_F = np.arange(len(flat_V)).reshape(-1, 3)
    flat_C = np.repeat(tri_col, 3, axis=0)        # one colour per duplicated vertex

    flat_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(flat_V),
        o3d.utility.Vector3iVector(flat_F))
    flat_mesh.vertex_colors = o3d.utility.Vector3dVector(flat_C.astype(np.float32)/255.0)
    flat_mesh.compute_vertex_normals()
    
    # rgb = np.zeros((len(V), 3), np.uint8)
    # rgb[:, 0] = (dense.astype(np.float32) * (255.0 / dense.max())).round().astype(np.uint8)
    # rgb[:, 1] = ((dense.max() - dense.astype(np.float32)) * (255.0 / dense.max())).round().astype(np.uint8)
    # rgb[:, 2] = make_random_blue_channel(dense, seed=30)
    # mesh = o3d.geometry.TriangleMesh(
    #     o3d.utility.Vector3dVector(V),
    #     o3d.utility.Vector3iVector(F))
    # mesh.vertex_colors = o3d.utility.Vector3dVector(rgb.astype(np.float32)/255.0)
    # mesh.compute_vertex_normals()

    # ---------- load COLMAP cameras/images ----------
    cams = read_cameras_binary(sparse_dir / "cameras.bin")
    imgs = read_images_binary (sparse_dir / "images.bin")

    # ---------- one off-screen renderer ----------
    w0, h0 = next(iter(cams.values()))["width"], next(iter(cams.values()))["height"]
    renderer = o3d.visualization.rendering.OffscreenRenderer(int(w0), int(h0))
    scene    = renderer.scene
    scene.set_background([0,0,0,0])

    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultUnlit"
    scene.add_geometry("semantic", flat_mesh, mat)

    # ---------- render each unique RGB file ----------

    # Print all available image names for debugging
    print("[INFO] Available image names in COLMAP images.bin:")
    for rec in imgs.values():
        print("  ", rec["name"])

    # Only process the image named 'DSC03423.JPG'
    for rec in imgs.values():
        fname = rec["name"]
        if fname != "DSC03423.JPG":
            continue

        cam  = cams[rec["cam_id"]]
        fx, fy, cx, cy = cam["params"]
        w,  h         = cam["width"], cam["height"]

        K = np.array([[fx, 0, cx],
                      [0,  fy, cy],
                      [0,  0,  1 ]], np.float64)
        scene.camera.set_projection(K, 0.01, 100.0, float(w), float(h))

        R = qvec_to_R(rec["qvec"]);  t = rec["tvec"]
        pos  = -(R.T @ t)
        look = pos + R.T @ np.array([0,0, 1])   # forward
        up   = R.T @ np.array([0,-1,0])         # COLMAP up vector

        # Debug output for DSC03423.JPG only
        near_plane = 0.2
        far_plane = 1.5
        img_w, img_h = int(w), int(h)
        corners = np.array([
            [0, 0],
            [img_w-1, 0],
            [img_w-1, img_h-1],
            [0, img_h-1]
        ], dtype=np.float32)
        def pixel_to_cam(x, y, fx, fy, cx, cy, depth):
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            return np.array([X, Y, Z])
        frustum_cam = [np.zeros(3)]
        for d in [near_plane, far_plane]:
            for x, y in corners:
                frustum_cam.append(pixel_to_cam(x, y, fx, fy, cx, cy, d))
        frustum_cam = np.stack(frustum_cam, axis=0)
        frustum_world = (R.T @ frustum_cam.T).T + pos
        view_dir = (look - pos)
        view_dir = view_dir / np.linalg.norm(view_dir)
        up_vec = up / np.linalg.norm(up)
        print("[DEBUG] Camera pose and frustum info:")
        print(f"fx, fy, cx, cy: {fx}, {fy}, {cx}, {cy}")
        print(f"Image size: {img_w}x{img_h}")
        print(f"near_plane: {near_plane}, far_plane: {far_plane}")
        print(f"Camera center (C): {pos}")
        print(f"View direction (world): {view_dir}")
        print(f"Up vector (world): {up_vec}")
        print(f"Frustum world points (first 5):\n{frustum_world[:5]}")

        scene.camera.look_at(look.tolist(), pos.tolist(), up.tolist())

        rgba = np.asarray(renderer.render_to_image())
        if rgba.shape[2] == 4:
            rgba[...,3] = 255

        out_path = out_dir / (Path(fname).stem + ".png")
        imageio.imwrite(out_path, rgba.astype(np.uint8))
        print("Rendered", out_path.name)
        break  # Only process DSC03423.JPG

    renderer = None
    print("✓ Finished rendering all masks →", out_dir.relative_to(scene_dir))
# ---------------------------------------------------------------------------


    renderer = None
    print("✓ Masks saved to", out_dir.relative_to(scene_dir))

# -------------------------------------------------------------------
if __name__ == "__main__":
    scene = Path("/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene")
    render_scene_masks(scene, scene/"semantic_mask")
