#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import re

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    # Robustly extract iteration numbers from checkpoint filenames, recursively
    import re
    saved_iters = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            # Match any sequence of digits before .ply or .pth at the end
            m = re.search(r'(\d+)(?=\.(ply|pth)$)', fname)
            if m:
                saved_iters.append(int(m.group(1)))
            # If file is named point_cloud.ply, extract digits from parent folder
            elif fname == 'point_cloud.ply':
                parent = os.path.basename(root)
                m2 = re.search(r'(\d+)', parent)
                if m2:
                    saved_iters.append(int(m2.group(1)))
    if not saved_iters:
        raise FileNotFoundError(f"No checkpoint files with iteration number found in {folder}")
    return max(saved_iters)
