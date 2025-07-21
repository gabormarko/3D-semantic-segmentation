import numpy as np
import argparse

def compute_confidence_stats(logits_path):
    logits = np.load(logits_path)  # shape: [num_classes, H, W]
    # Sort logits along class axis for each pixel
    sorted_logits = np.sort(logits, axis=0)
    # Confidence = top1 - top2 logit per pixel
    confidences = sorted_logits[-1] - sorted_logits[-2]  # shape: [H, W]
    flat = confidences.flatten()
    stats = {
        'min': float(flat.min()),
        'max': float(flat.max()),
        'mean': float(flat.mean()),
        'std': float(flat.std()),
    }
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics of pixel label confidence from composited logit .npy file.")
    parser.add_argument('--logits', type=str, required=True, help='Path to composited logits .npy file')
    args = parser.parse_args()

    stats = compute_confidence_stats(args.logits)
    print(f"Confidence statistics (top1 - top2 logit) for {args.logits}:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
