# Copy of voxeltoGaussian.py for logit-based semantic pipeline
# See original for full docstring and CLI

#!/usr/bin/env python3
# open_vocab_pipeline.py
#
# Open-Vocabulary 3-D Labeller
#  – voxels:  .pt  **or**  .npz   (keys: 'pos', 'feat')
#  – gaussians: .pt / .pth  **or**  .npz   (key: 'mu')
#
# Pipeline
#   1. convert   : (optional)  .pt  →  .npz   for voxels
#   2. build_map : Gaussian → nearest-voxel index (k = 1)
#   3. query     : prompt → per-Gaussian binary labels
# ---------------------------------------------------------------------------


import argparse
import pathlib
import gc
import numpy as np
import torch
from sklearn.neighbors import KDTree
from transformers import CLIPModel, CLIPTokenizerFast
from typing import Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def convert_pt_to_npz(pt_path: pathlib.Path, npz_path: pathlib.Path) -> None:
    """Re-serialise a voxel .pt dump into compressed .npz."""
    data = torch.load(pt_path, map_location="cpu")
    np.savez_compressed(
        npz_path,
        pos=data["xyz"].cpu().float().numpy(),
        feat=data["avg_feats"].cpu().float().numpy(),
    )

def load_voxels(path: pathlib.Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (pos, feat) as *CPU* float tensors, regardless of source type."""
    if path.suffix in {".pt", ".pth"}:
        d = torch.load(path, map_location="cpu")
        # Use 'xyz' and 'avg_feats' keys as produced by aggregate_voxel_features_onthefly.py
        if "xyz" in d and "avg_feats" in d:
            return d["xyz"].float(), d["avg_feats"].float()
        elif "pos" in d and "feat" in d:
            return d["pos"].float(), d["feat"].float()
        else:
            raise KeyError(f"Voxel file {path} missing required keys. Found: {list(d.keys())}")
    elif path.suffix == ".npz":
        d = np.load(path)
        return torch.from_numpy(d["pos"]).float(), torch.from_numpy(d["feat"]).float()
    raise ValueError(f"Unsupported voxel file format: {path}")

def load_gaussians_mu(path: pathlib.Path) -> torch.Tensor:
    """Load Gaussian means from .pth/.pt or .npy/.npz file."""
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        # Try to access [0][1] (mu) as in old format
        try:
            mu = data[0][1]
        except Exception:
            # If not, try 'mu' key
            mu = data["mu"] if "mu" in data else None
        if mu is None:
            raise KeyError(f"Could not find Gaussian centers in {path}. Found keys: {list(data.keys())}")
        return mu.detach().cpu().float()
    elif path.suffix in {".npy", ".npz"}:
        data = np.load(path)
        # If npz, look for 'mu' or use first array
        if hasattr(data, "files") and "mu" in data.files:
            mu = data["mu"]
        elif hasattr(data, "files") and len(data.files) > 0:
            mu = data[data.files[0]]
        else:
            mu = data
        return torch.from_numpy(mu).float()
    else:
        raise ValueError(f"Unsupported Gaussian file format: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Mapping stage
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def map_gaussians_to_voxels(
    voxel_pos: torch.Tensor,  # (N,3)  cpu
    gaussian_mu: torch.Tensor,  # (M,3) cpu
    batch_size: int = 200_000,
) -> torch.LongTensor:
    """
    For each Gaussian centre, find index of its 1-NN voxel.
    Returns (M,) int64 tensor  (still on CPU).
    """
    tree = KDTree(voxel_pos.numpy(), leaf_size=16)
    idx_chunks = []
    M = gaussian_mu.shape[0]
    for s in range(0, M, batch_size):
        e = min(s + batch_size, M)
        idx = tree.query(
            gaussian_mu[s:e].numpy(), k=1, return_distance=False
        )  # (batch,1)
        idx_chunks.append(torch.from_numpy(idx[:, 0]))
    return torch.cat(idx_chunks, dim=0)

# ─────────────────────────────────────────────────────────────────────────────
# Query stage
# ─────────────────────────────────────────────────────────────────────────────

# @torch.inference_mode()
# def prompt_to_gaussian_labels(
#     prompt: str,
#     voxel_feat: torch.Tensor,      # (N,512) cpu or gpu
#     g2v_idx: torch.LongTensor,     # (M,)    cpu or gpu
#     device: str = "cuda",
# ) -> np.ndarray:
#     tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

#     # Encode text
#     tokens = tokenizer([prompt], return_tensors="pt").to(device)
#     q = model.get_text_features(**tokens)
#     q = torch.nn.functional.normalize(q, dim=1)  # (1,512)

#     # Voxel-level similarities
#     vf = torch.nn.functional.normalize(voxel_feat.to(device), dim=1)  # (N,512)
#     sim = (vf @ q.T).squeeze(1)  # (N,)

#     # One-hot semantic field on voxels
#     v_sem = torch.zeros_like(sim)
#     v_sem[sim.argmax()] = 1.0

#     # Gather onto Gaussians
#     g_sem = v_sem[g2v_idx.to(device)]       # (M,)
#     return (g_sem > 0.5).to(torch.uint16).cpu().numpy()

@torch.inference_mode()
def prompts_to_gaussian_onehot(
    prompt: List[str],               # list of P prompt strings
    voxel_feat: torch.Tensor,         # (N,512)  CPU or GPU
    g2v_idx: torch.LongTensor,        # (M,)     CPU or GPU
    device: str = "cuda",
) -> np.ndarray:
    """
    Projects raw voxel features using LSeg model, then assigns each voxel its best label.
    Transfers voxel labels to Gaussians via nearest neighbor mapping.
    """
    # --- Original direct CLIP comparison (incorrect for raw features) ---
    # tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    # model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    # tok = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    # text_emb = model.get_text_features(**tok)                # (P,512)
    # text_emb = torch.nn.functional.normalize(text_emb, dim=1)
    # vf = torch.nn.functional.normalize(voxel_feat.to(device), dim=1)  # (N,512)
    # sim = vf @ text_emb.T                                             # (N,P)  fp32 or fp16
    # voxel_cls = sim.argmax(dim=1).to(torch.int16)                     # (N,)
    # gauss_cls = voxel_cls[g2v_idx.to(device)]
    # return gauss_cls.cpu().numpy()# (M,)

    # --- Correct logic: project features using LSeg ---
    import sys
    sys.path.append("/home/neural_fields/Unified-Lift-Gabor/lang-seg")
    from modules.models.lseg_net import LSeg
    # Local subclass to add labels for open-vocab query only
    class LSegWithLabels(LSeg):
        def __init__(self, *args, labels=None, **kwargs):
            self.labels = labels
            super().__init__(*args, **kwargs)
            self.text = clip.tokenize(self.labels)

    import clip
    checkpoint_path = "/home/neural_fields/Unified-Lift-Gabor/lang-seg/checkpoints/demo_e200.ckpt"
    head = torch.nn.Identity()  # Use identity if you don't need output_conv
    model = LSegWithLabels(head=head, features=256, backbone="clip_vitl16_384", arch_option=0, block_depth=0, activation='lrelu', labels=prompt)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)

    N, C = voxel_feat.shape
    num_labels = len(prompt)
    print(f"[DEBUG] Loaded voxel features: {N} voxels, {C} feature dim")
    print(f"[DEBUG] Using {num_labels} labels: {prompt[:5]}{'...' if num_labels > 5 else ''}")

    # Project features in batches for memory efficiency
    batch_size = 10000
    voxel_logits = torch.empty((N, num_labels), dtype=torch.float32, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_feat = voxel_feat[start:end].to(device)
        batch_feat = batch_feat.unsqueeze(-1).unsqueeze(-1)  # [batch, C, 1, 1]
        with torch.no_grad():
            batch_logits = model.project_features_to_labels(batch_feat, labelset=prompt, device=device)
            batch_logits = batch_logits.squeeze(-1).squeeze(-1)  # [batch, num_labels]
        voxel_logits[start:end] = batch_logits
        print(f"[DEBUG] Projected voxels {start}-{end}, logits shape: {batch_logits.shape}")

    # Assign each voxel its best label
    voxel_cls = voxel_logits.argmax(dim=1).to(torch.int16)  # (N,)
    print(f"[DEBUG] Voxel label assignment done. Example labels: {voxel_cls[:10].tolist()}")

    # Propagate to Gaussians
    gauss_cls = voxel_cls[g2v_idx.to(device)]  # (M,)
    gauss_logits = voxel_logits[g2v_idx.to(device)]  # (M, num_labels)
    print(f"[DEBUG] Propagated labels and logits to {gauss_cls.shape[0]} Gaussians. Example labels: {gauss_cls[:10].tolist()}")

    return gauss_cls.cpu().numpy(), gauss_logits.cpu().numpy()  # (M,), (M,num_labels)

# ─────────────────────────────────────────────────────────────────────────────
# CLI wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _cli_convert(args: argparse.Namespace) -> None:
    convert_pt_to_npz(args.pt, args.out)
    print(f"[✓] Converted {args.pt} → {args.out}")

def _cli_build_map(args: argparse.Namespace) -> None:
    voxel_pos, _ = load_voxels(args.vox)
    gaussian_mu = load_gaussians_mu(args.gauss)
    idx = map_gaussians_to_voxels(voxel_pos, gaussian_mu, batch_size=args.batch)
    np.save(args.out, idx.numpy())
    print(f"[✓] Map saved: {args.out}  shape={idx.shape}")

def _cli_query(args: argparse.Namespace) -> None:
    _, voxel_feat = load_voxels(args.vox)
    g2v_idx = torch.from_numpy(np.load(args.map))
    labels, logits = prompts_to_gaussian_onehot(
        args.prompt, voxel_feat, g2v_idx, device=args.device
    )
    # Save both label indices, logits, and prompt list in output
    np.savez(args.out, labels=labels, logits=logits, prompts=np.array(args.prompt))
    print(f"[✓] Labels, logits, and prompts saved: {args.out}  positives={np.sum(labels)}/{labels.size}")

    # --- Summary statistics ---

    print("\n[SUMMARY] Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for i, c in zip(unique, counts):
        label_name = args.prompt[i] if i < len(args.prompt) else f"Label {i}"
        print(f"  {label_name:20s} (idx={i}): count={c}")

    # Save histogram PNG for assigned labels
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.bar([args.prompt[i] if i < len(args.prompt) else f"Label {i}" for i in unique], counts, color='skyblue')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Gaussian Assigned Label Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        hist_path = str(args.out).replace('.npz', '_label_hist.png')
        plt.savefig(hist_path)
        plt.close()
        print(f"[✓] Saved label histogram PNG: {hist_path}")
    except Exception as e:
        print(f"[WARN] Could not save label histogram PNG: {e}")

    print("\n[DEBUG] Argmax label histogram for Gaussians:")
    argmax_labels = np.argmax(logits, axis=1)
    uniq, cnts = np.unique(argmax_labels, return_counts=True)
    for i, c in zip(uniq, cnts):
        label_name = args.prompt[i] if i < len(args.prompt) else f"Label {i}"
        print(f"  {label_name:20s} (idx={i}): count={c}")

    # Save histogram PNG for argmax labels
    try:
        plt.figure(figsize=(8,4))
        plt.bar([args.prompt[i] if i < len(args.prompt) else f"Label {i}" for i in uniq], cnts, color='salmon')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Gaussian Argmax Label Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        hist_path2 = str(args.out).replace('.npz', '_argmax_hist.png')
        plt.savefig(hist_path2)
        plt.close()
        print(f"[✓] Saved argmax label histogram PNG: {hist_path2}")
    except Exception as e:
        print(f"[WARN] Could not save argmax label histogram PNG: {e}")

    print("\n[DEBUG] Example Gaussian logits (first 5):")
    for i in range(min(5, logits.shape[0])):
        print(f"  Gaussian {i}: logits={logits[i]}, argmax={np.argmax(logits[i])}")

    print("\n[SUMMARY] Logit statistics per class:")
    for i, label_name in enumerate(args.prompt):
        vals = logits[:, i]
        print(f"  {label_name:20s} (idx={i}): min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f} std={vals.std():.4f} count={vals.size}")

    # --- Save colored .ply for Gaussians ---
    # Load Gaussian centers
    if not hasattr(args, 'gauss'):
        raise ValueError("Missing required argument --gauss for Gaussian centers. Please add --gauss <path> to your query command.")
    gaussian_mu = load_gaussians_mu(args.gauss)
    num_labels = len(args.prompt)
    def get_palette(num_cls):
        n = num_cls
        palette = [0]*(n*3)
        for j in range(n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while lab > 0:
                palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i += 1
                lab >>= 3
        return palette

    palette = get_palette(num_labels)
    # Assign color to each Gaussian
    gaussian_colors = np.zeros((gaussian_mu.shape[0], 3), dtype=np.uint8)
    for i in range(gaussian_mu.shape[0]):
        idx = int(argmax_labels[i])
        gaussian_colors[i, 0] = palette[3*idx+0]
        gaussian_colors[i, 1] = palette[3*idx+1]
        gaussian_colors[i, 2] = palette[3*idx+2]

    ply_path = str(args.out).replace('.npz', '_colored_gaussians.ply')
    with open(ply_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {gaussian_mu.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(gaussian_mu.shape[0]):
            x, y, z = gaussian_mu[i].tolist()
            r, g, b = gaussian_colors[i].tolist()
            f.write(f'{x} {y} {z} {r} {g} {b}\n')
    print(f"[✓] Colored Gaussian .ply saved: {ply_path}")

    # Free VRAM if we were on GPU
    if args.device == "cuda":
        del voxel_feat, g2v_idx
        torch.cuda.empty_cache()
    gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser("Open-Vocabulary 3-D Labeller")
    sp = p.add_subparsers(dest="cmd", required=True)

    cvt = sp.add_parser("convert", help=".pt/.pth → .npz (voxels)")
    cvt.add_argument("--pt", type=pathlib.Path, required=True)
    cvt.add_argument("--out", type=pathlib.Path, required=True)
    cvt.set_defaults(func=_cli_convert)

    bld = sp.add_parser("build_map", help="Gaussian → voxel 1-NN index map")
    bld.add_argument("--vox", type=pathlib.Path, required=True,
                     help=".pt/.npz voxel file")
    bld.add_argument("--gauss", type=pathlib.Path, required=True,
                     help=".pt/.pth or .npz with 'mu'")
    bld.add_argument("--out", type=pathlib.Path, required=True,
                     help=".npy output of shape (M,)")
    bld.add_argument("--batch", type=int, default=200_000)
    bld.set_defaults(func=_cli_build_map)

    qry = sp.add_parser("query", help="Prompt → per-Gaussian labels")
    qry.add_argument("--vox", type=pathlib.Path, required=True)
    qry.add_argument("--map", type=pathlib.Path, required=True,
                     help=".npy produced by build_map")
    qry.add_argument("--gauss", type=pathlib.Path, required=True,
                     help=".pt/.pth/.npz/.npy file with Gaussian centers (mu)")
    qry.add_argument("--prompt", type=str, nargs="+", required=True,
                 help="List of prompt categories, e.g. --prompt 'chair' 'table' 'lamp'")
    qry.add_argument("--out", type=pathlib.Path, required=True,
                     help=".npy binary labels")
    qry.add_argument("--device", type=str, default="cuda",
                     choices=["cuda", "cpu"])
    qry.set_defaults(func=_cli_query)

    args = p.parse_args()
    args.func(args)
