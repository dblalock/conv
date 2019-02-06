//
//  catconv.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "catconv.hpp"

using namespace tensorflow;

// TODO allow more than just floats:
//  -see https://www.tensorflow.org/guide/extend/op#type_polymorphism
REGISTER_OP("CatConv")
    .Input("input: float")
    .Input("filter: float")
    .Output("activations: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class CatConvOp : public OpKernel {
public:
    explicit CatConvOp(OpKernelConstruction* context) : OpKernel(context) {}

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

REGISTER_KERNEL_BUILDER(Name("CatConv").Device(DEVICE_CPU), CatConvOp);
