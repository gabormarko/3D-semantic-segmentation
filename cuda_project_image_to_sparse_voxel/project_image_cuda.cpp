#include <torch/extension.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

///////// 2D - > 3D /////////
// CUDA forward declaration
void project_features_cuda_forward_impl(at::Tensor encoded_2d_features,
                                        at::Tensor occupancy_3D,
                                        at::Tensor viewMatrixInv,
                                        at::Tensor intrinsicParams,
                                        at::Tensor opts,
                                        at::Tensor mapping2dto3d_num,
                                        at::Tensor projected_features,
                                        at::Tensor pred_mode_t,
                                        at::Tensor grid_origin,      // <-- add this
                                        float voxel_size);             // <-- add this);

// C++ wrapper with input validation
void project_features_cuda_forward(at::Tensor encoded_2d_features,
                                   at::Tensor occupancy_3D,
                                   at::Tensor viewMatrixInv,
                                   at::Tensor intrinsicParams,
                                   at::Tensor opts,
                                   at::Tensor mapping2dto3d_num,
                                   at::Tensor projected_features,
                                   at::Tensor pred_mode_t,
                                   at::Tensor grid_origin,      // <-- add this
                                   float voxel_size             // <-- add this
                                ) 
{
    std::cout << "Entering C++ wrapper: project_features_cuda_forward" << std::endl;

    // Device and contiguity checks
    CHECK_INPUT(encoded_2d_features);
    CHECK_INPUT(occupancy_3D);
    CHECK_INPUT(viewMatrixInv);
    CHECK_INPUT(intrinsicParams);
    CHECK_INPUT(mapping2dto3d_num);
    CHECK_INPUT(projected_features);

    // Dtype checks
    TORCH_CHECK(encoded_2d_features.scalar_type() == at::kFloat, "encoded_2d_features must be float32");
    TORCH_CHECK(occupancy_3D.scalar_type() == at::kLong, "occupancy_3D must be int64");
    TORCH_CHECK(viewMatrixInv.scalar_type() == at::kFloat, "viewMatrixInv must be float32");
    TORCH_CHECK(intrinsicParams.scalar_type() == at::kFloat, "intrinsicParams must be float32");
    TORCH_CHECK(opts.scalar_type() == at::kFloat, "opts must be float32");
    TORCH_CHECK(mapping2dto3d_num.scalar_type() == at::kInt, "mapping2dto3d_num must be int32");
    TORCH_CHECK(projected_features.scalar_type() == at::kFloat, "projected_features must be float32");
    TORCH_CHECK(pred_mode_t.scalar_type() == at::kBool, "pred_mode_t must be bool");

    // Shape checks
    TORCH_CHECK(encoded_2d_features.dim() == 5, "encoded_2d_features must be 5D [B,V,H,W,C]");
    TORCH_CHECK(occupancy_3D.dim() == 4, "occupancy_3D must be 4D [B,Z,Y,X]");
    TORCH_CHECK(viewMatrixInv.dim() == 1, "viewMatrixInv must be 1D flattened");
    TORCH_CHECK(intrinsicParams.dim() == 2, "intrinsicParams must be 2D [B,4]");
    TORCH_CHECK(opts.dim() == 1 && opts.numel() == 5, "opts must be 1D with 5 elements");
    TORCH_CHECK(pred_mode_t.dim() == 1 && pred_mode_t.numel() == 1, "pred_mode_t must be scalar");

    // Call the actual CUDA implementation
    project_features_cuda_forward_impl(
        encoded_2d_features,
        occupancy_3D, 
        viewMatrixInv,
        intrinsicParams,
        opts,
        mapping2dto3d_num,
        projected_features,
        pred_mode_t,
        grid_origin,      // <-- add this
        voxel_size        // <-- add this
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_features_cuda", &project_features_cuda_forward, "Projecting from 2D to 3D (With CUDA kernels)");
    // m.def("unproject_depth_images", &unproject_depth_images, "Projecting from 2D to 3D a batch of depth images with camera poses and parameters");
}
