#!/usr/bin/env python3
"""
Analyze gaussian_semantics.npz file: print stats, label distribution, RGB values, and save histograms as PNGs.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main(npz_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path)
    labels = data['labels']
    colors = data['colors']

    print(f"Loaded: {npz_path}")
    print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"colors shape: {colors.shape}, dtype: {colors.dtype}")

    # Label stats
    min_label = labels.min()
    max_label = labels.max()
    print(f"Label min: {min_label}, max: {max_label}")
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} Gaussians")

    # Save label histogram
    plt.figure()
    plt.bar(unique, counts, color='skyblue')
    plt.xlabel('Label Index')
    plt.ylabel('Number of Gaussians')
    plt.title('Gaussian Label Distribution')
    plt.savefig(os.path.join(out_dir, 'label_histogram.png'))
    plt.close()

    # RGB stats
    print(f"RGB min: {colors.min(axis=0)}, max: {colors.max(axis=0)}")
    print(f"First 5 RGB values:")
    print(colors[:5])

    # Save RGB histograms
    for i, channel in enumerate(['R', 'G', 'B']):
        plt.figure()
        plt.hist(colors[:, i], bins=32, color=channel.lower(), alpha=0.7)
        plt.xlabel(f'{channel} value')
        plt.ylabel('Count')
        plt.title(f'{channel} Channel Histogram')
        plt.savefig(os.path.join(out_dir, f'{channel}_histogram.png'))
        plt.close()

    # Compute mean RGB per label
    means = np.array([colors[labels == u].mean(axis=0) for u in unique])

    print("\nMean RGB value for each label:")
    # Try to get label names from npz file ('prompts'), else environment or argument
    label_names = None
    if 'prompts' in data:
        label_names = [str(x) for x in data['prompts']]
    else:
        try:
            label_names_env = os.environ.get('GAUSSIAN_LABEL_NAMES', None)
            if label_names_env:
                label_names = label_names_env.split(',')
        except Exception:
            pass
    # If label_names not set, just use index
    for idx, u in enumerate(unique):
        mean_rgb = means[idx]
        count = counts[idx]
        label_str = f"Label {u}"
        if label_names and idx < len(label_names):
            label_str = f"{label_names[idx]}, Label {u}"
        print(f"  {label_str}, R={mean_rgb[0]:.2f}, G={mean_rgb[1]:.2f}, B={mean_rgb[2]:.2f}, count={count}")

    plt.figure(figsize=(8,4))
    for i, channel in enumerate(['R', 'G', 'B']):
        plt.bar(unique + i*0.25, means[:, i], width=0.25, label=channel)
    plt.xlabel('Label Index')
    plt.ylabel('Mean RGB Value')
    plt.title('Mean RGB per Label')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'mean_rgb_per_label.png'))
    plt.close()

    # Create a legend PNG: label name (index, count) and color
    legend_height = 40 * len(unique)
    legend_width = 500
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(legend_width/100, legend_height/100))
    ax.set_xlim(0, legend_width)
    ax.set_ylim(0, legend_height)
    ax.axis('off')
    for idx, u in enumerate(unique):
        mean_rgb = means[idx]
        count = counts[idx]
        label_str = f"Label {u}"
        if label_names and idx < len(label_names):
            label_str = f"{label_names[idx]} (Label {u}, count={count})"
        else:
            label_str = f"Label {u} (count={count})"
        y = legend_height - 40 * (idx + 1)
        # Draw color rectangle
        rect = mpatches.Rectangle((10, y), 30, 30, color=mean_rgb/255.0, ec='black')
        ax.add_patch(rect)
        # Draw text
        ax.text(50, y+15, label_str, va='center', ha='left', fontsize=14)
    plt.savefig(os.path.join(out_dir, 'label_legend.png'), bbox_inches='tight')
    plt.close()
    print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"colors shape: {colors.shape}, dtype: {colors.dtype}")

    # Label stats
    min_label = labels.min()
    max_label = labels.max()
    print(f"Label min: {min_label}, max: {max_label}")
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} Gaussians")

    # Save label histogram
    plt.figure()
    plt.bar(unique, counts, color='skyblue')
    plt.xlabel('Label Index')
    plt.ylabel('Number of Gaussians')
    plt.title('Gaussian Label Distribution')
    plt.savefig(os.path.join(out_dir, 'label_histogram.png'))
    plt.close()

    # RGB stats
    print(f"RGB min: {colors.min(axis=0)}, max: {colors.max(axis=0)}")
    print(f"First 5 RGB values:")
    print(colors[:5])

    # Save RGB histograms
    for i, channel in enumerate(['R', 'G', 'B']):
        plt.figure()
        plt.hist(colors[:, i], bins=32, color=channel.lower(), alpha=0.7)
        plt.xlabel(f'{channel} value')
        plt.ylabel('Count')
        plt.title(f'{channel} Channel Histogram')
        plt.savefig(os.path.join(out_dir, f'{channel}_histogram.png'))
        plt.close()



    # Compute mean RGB per label
    means = np.array([colors[labels == u].mean(axis=0) for u in unique])

    print("\nMean RGB value for each label:")
    # Try to get label names from environment or argument
    label_names = None
    try:
        #import os
        label_names_env = os.environ.get('GAUSSIAN_LABEL_NAMES', None)
        if label_names_env:
            label_names = label_names_env.split(',')
    except Exception:
        pass
    # If label_names not set, just use index
    for idx, u in enumerate(unique):
        mean_rgb = means[idx]
        count = counts[idx]
        label_str = f"Label {u}"
        if label_names and idx < len(label_names):
            label_str = f"{label_names[idx]}, Label {u}"
        print(f"  {label_str}, R={mean_rgb[0]:.2f}, G={mean_rgb[1]:.2f}, B={mean_rgb[2]:.2f}, count={count}")

    plt.figure(figsize=(8,4))
    for i, channel in enumerate(['R', 'G', 'B']):
        plt.bar(unique + i*0.25, means[:, i], width=0.25, label=channel)
    plt.xlabel('Label Index')
    plt.ylabel('Mean RGB Value')
    plt.title('Mean RGB per Label')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'mean_rgb_per_label.png'))
    plt.close()

    print(f"Histograms saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze gaussian_semantics.npz file")
    parser.add_argument('--npz', type=str, required=True, help='Path to gaussian_semantics.npz')
    parser.add_argument('--out_dir', type=str, required=False, default='gaussian_semantics_stats', help='Output directory for histograms')
    parser.add_argument('--label_names', type=str, nargs='*', help='Optional list of label names, in order')
    args = parser.parse_args()
    # Optionally set label names as environment variable for use in main
    if args.label_names:
        import os
        os.environ['GAUSSIAN_LABEL_NAMES'] = ','.join(args.label_names)
    main(args.npz, args.out_dir)
