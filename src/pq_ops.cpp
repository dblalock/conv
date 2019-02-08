//
//  pq_ops.cpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "catconv.hpp"

using namespace tensorflow;

// TODO allow more than just floats:
//  -see https://www.tensorflow.org/guide/extend/op#type_polymorphism
REGISTER_OP("PqConv")
    .Input("input: float")
    .Input("centroids: float")
    .Output("assignments: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class PqConvOp : public OpKernel {
public:
    explicit PqConvOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& img = context->input(0);
        const Tensor& filter = context->input(1);

        auto in_shape = img.shape().dim_sizes();
        auto filt_shape = filter.shape().dim_sizes();

        DCHECK_EQ(img.shape().dims(), 4);
        DCHECK_EQ(filter.shape().dims(), 4);
        DCHECK_EQ(filt_shape[1], in_shape[1]);  // same number of channels

        TensorShape out_tfshape = img.shape();
        auto nout = filt_shape[0];
        out_tfshape.set_dim(1, nout);

        Tensor* out_container = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, out_tfshape, &out_container));

        // TODO template this class
        using activationT = int32_t;
        using coefT = float;

        auto in = img.tensor<activationT, 4>();
        auto filt = filter.tensor<coefT, 4>();
        auto out = out_container->tensor<activationT, 4>();

        cat2cat_conv2d_nchw_x_gvchw_valid(
            in.data(), in_shape[0], in_shape[1], in_shape[2], in_shape[3],
            filt.data(), filt_shape[0], filt_shape[1], filt_shape[2], filt_shape[3],
            out.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("PqConv").Device(DEVICE_CPU), PqConvOp);
