
# 1. Set working directories
mkdir -p workspace_7b898d2d22
cd workspace_7b898d2d22
mkdir -p images sparse

# 2. Copy all undistorted images into `images/`
cp -r ../../data/7b898d2d22/dslr/resized_undistorted_images/* images/

# 3. Extract features with PINHOLE camera model (no GUI)
DISPLAY="" colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0

# 4. Match features (no GUI)
DISPLAY="" colmap exhaustive_matcher \
    --database_path database.db \
    --SiftMatching.use_gpu 0

# 5. Reconstruct sparse model
colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse \
    --Mapper.num_threads 1
