//
//  exampe_ops.cpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


class ZeroOutOp : public OpKernel {
public:
    explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        // const Tensor& input_tensor = context->mutable_input(0);
        auto input = input_tensor.flat<int32>();

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, input_tensor.shape(), &output_tensor));

        auto output_flat = output_tensor->flat<int32>();

        // set all but 1st output element to 0
        int N = input.size();
        for (int i = 1; i < N; i++) {
            output_flat(i) = 0;
            // input_tensor(i) = 7;
            // input_tensor.data()[i] = 0;
        }
        if (N > 0) {
            output_flat(0) = input(0);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);

// ================================================================ 2D float

REGISTER_OP("DummyFloat")
    .Input("input: float")
    .Input("filter: float")
    .Output("activations: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class DummyFloatOp : public OpKernel {
public:
    explicit DummyFloatOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& img = context->input(0);
        const Tensor& filter = context->input(1);
        // const int nout = context->input(2);

        TensorShape out_shape = img.shape();
        TensorShape filt_shape = img.shape();
        auto filt_dims = filter.shape().dim_sizes();
        auto n = out_shape.dim_sizes()[0];
        auto nout = filt_dims[0];

        out_shape.set_dim(1, nout);

        Tensor* out_container = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, out_shape, &out_container));

        // get an Eigen Tensor (I think?) from our TF tensor
        auto out = out_container->tensor<float, 2>();

        // TODO call actual function once this compiles and runs
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nout; j++) {
                out(i, j) = nout * i + j;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("DummyFloat").Device(DEVICE_CPU), DummyFloatOp);

