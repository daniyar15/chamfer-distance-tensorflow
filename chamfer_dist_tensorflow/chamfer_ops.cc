/*
 * chamfer_ops.cc
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/platform/status.h"

using namespace tensorflow;

// Forward declaration of the kernel launchers
void ChamferDistKernelLauncher(int b, int n, const float* xyz,
                               int m, const float* xyz2,
                               float* result, int* result_i, float* result2, int* result2_i,
                               const Eigen::GpuDevice& d);

void ChamferDistGradKernelLauncher(int b, int n, const float* xyz1,
                                   int m, const float* xyz2,
                                   const float* grad_dist1, const int* idx1,
                                   const float* grad_dist2, const int* idx2,
                                   float* grad_xyz1, float* grad_xyz2,
                                   const Eigen::GpuDevice& d);

// --------------------------------------------------------------------------
// 1. Forward Op Registration
// --------------------------------------------------------------------------
REGISTER_OP("ChamferDist")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("dist1: float32")
    .Output("dist2: float32")
    .Output("idx1: int32")
    .Output("idx2: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // Output shapes: dist1/idx1 are [B, N], dist2/idx2 are [B, M]
        ::tensorflow::shape_inference::ShapeHandle batch;
        ::tensorflow::shape_inference::ShapeHandle n_points;
        ::tensorflow::shape_inference::ShapeHandle m_points;
        
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &batch));
        
        // xyz1 shape is [Batch, N, 3]
        ::tensorflow::shape_inference::ShapeHandle xyz1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &xyz1_shape));
        ::tensorflow::shape_inference::DimensionHandle b_dim = c->Dim(xyz1_shape, 0);
        ::tensorflow::shape_inference::DimensionHandle n_dim = c->Dim(xyz1_shape, 1);
        
        // xyz2 shape is [Batch, M, 3]
        ::tensorflow::shape_inference::ShapeHandle xyz2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &xyz2_shape));
        ::tensorflow::shape_inference::DimensionHandle m_dim = c->Dim(xyz2_shape, 1);

        // Set outputs
        c->set_output(0, c->MakeShape({b_dim, n_dim}));
        c->set_output(1, c->MakeShape({b_dim, m_dim}));
        c->set_output(2, c->MakeShape({b_dim, n_dim}));
        c->set_output(3, c->MakeShape({b_dim, m_dim}));
        
        return OkStatus();
    });

class ChamferDistOp : public OpKernel {
public:
    explicit ChamferDistOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz1_tensor = context->input(0);
        const Tensor& xyz2_tensor = context->input(1);

        int b = xyz1_tensor.shape().dim_size(0);
        int n = xyz1_tensor.shape().dim_size(1);
        int m = xyz2_tensor.shape().dim_size(1);

        // Allocate outputs
        Tensor* dist1_tensor = nullptr;
        Tensor* dist2_tensor = nullptr;
        Tensor* idx1_tensor = nullptr;
        Tensor* idx2_tensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n}, &dist1_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b, m}, &dist2_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b, n}, &idx1_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{b, m}, &idx2_tensor));

        // Launch Kernel
        auto xyz1_flat = xyz1_tensor.flat<float>();
        auto xyz2_flat = xyz2_tensor.flat<float>();
        auto dist1_flat = dist1_tensor->flat<float>();
        auto dist2_flat = dist2_tensor->flat<float>();
        auto idx1_flat = idx1_tensor->flat<int>();
        auto idx2_flat = idx2_tensor->flat<int>();

        ChamferDistKernelLauncher(b, n, xyz1_flat.data(),
                                  m, xyz2_flat.data(),
                                  dist1_flat.data(), idx1_flat.data(),
                                  dist2_flat.data(), idx2_flat.data(),
                                  context->eigen_device<Eigen::GpuDevice>());
    }
};
REGISTER_KERNEL_BUILDER(Name("ChamferDist").Device(DEVICE_GPU), ChamferDistOp);


// --------------------------------------------------------------------------
// 2. Gradient Op Registration
// --------------------------------------------------------------------------
REGISTER_OP("ChamferDistGrad")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Input("idx1: int32")
    .Input("idx2: int32")
    .Input("grad_dist1: float32")
    .Input("grad_dist2: float32")
    .Output("grad_xyz1: float32")
    .Output("grad_xyz2: float32");

class ChamferDistGradOp : public OpKernel {
public:
    explicit ChamferDistGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz1 = context->input(0);
        const Tensor& xyz2 = context->input(1);
        const Tensor& idx1 = context->input(2);
        const Tensor& idx2 = context->input(3);
        const Tensor& grad_dist1 = context->input(4);
        const Tensor& grad_dist2 = context->input(5);

        int b = xyz1.shape().dim_size(0);
        int n = xyz1.shape().dim_size(1);
        int m = xyz2.shape().dim_size(1);

        Tensor* grad_xyz1 = nullptr;
        Tensor* grad_xyz2 = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 3}, &grad_xyz1));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b, m, 3}, &grad_xyz2));

        ChamferDistGradKernelLauncher(b, n, xyz1.flat<float>().data(),
                                      m, xyz2.flat<float>().data(),
                                      grad_dist1.flat<float>().data(), idx1.flat<int>().data(),
                                      grad_dist2.flat<float>().data(), idx2.flat<int>().data(),
                                      grad_xyz1->flat<float>().data(),
                                      grad_xyz2->flat<float>().data(),
                                      context->eigen_device<Eigen::GpuDevice>());
    }
};
REGISTER_KERNEL_BUILDER(Name("ChamferDistGrad").Device(DEVICE_GPU), ChamferDistGradOp);