#!/usr/bin/env python3
"""
Generate camera_params.json and scaled_camera_params.json in one go.
Usage:
  python generate_scaled_camera_params.py \
      --sparse_dir /path/to/colmap/sparse \
      --downscale_factor 4 \
      --output_dir camera_params
This will run the existing extract and scale scripts to populate:
  ${output_dir}/camera_params.json
  ${output_dir}/scaled_camera_params.json
"""
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract and scale COLMAP camera parameters.")
    parser.add_argument('--sparse_dir', required=True, help='COLMAP sparse reconstruction directory')
    parser.add_argument('--downscale_factor', type=int, required=True, help='LSeg downscale factor')
    parser.add_argument('--output_dir', required=True, help='Directory to write camera_params.json and scaled_camera_params.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Extract raw camera parameters
    cmd1 = [sys.executable, '../cuda_project_clean/extract_colmap_cameras.py',
            '--sparse_dir', args.sparse_dir,
            '--output_dir', args.output_dir]
    print("Running:", ' '.join(cmd1))
    subprocess.run(cmd1, check=True)

    # Step 2: Scale for LSeg features
    raw_json = os.path.join(args.output_dir, 'camera_params.json')
    cmd2 = [sys.executable, '../cuda-debug/calculate_scaled_camera_params.py',
            '--camera_params_path', raw_json,
            '--downscale_factor', str(args.downscale_factor),
            '--output_dir', args.output_dir]
    print("Running:", ' '.join(cmd2))
    subprocess.run(cmd2, check=True)

    print(f"Generated camera_params.json and scaled_camera_params.json in {args.output_dir}")

if __name__ == '__main__':
    main()
