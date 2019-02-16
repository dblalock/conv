//
//  ksparse_interop_grad.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _ksparse_interop_grad_hpp
#define _ksparse_interop_grad_hpp

#include <assert.h>

#include "arrayview.hpp"
#include "ksparse_interop.hpp"
#include "ksparse_util.hpp"

// packed errs from output x group_sz -> dense grads for input
// conveniently, this is just a call to sparse2dense()
template<class Data>
static const void dense2sparse_nhwc_grad(const DataT* errs_data, int nimgs,
    int img_nrows, int img_ncols, int img_ngroups, uint8_t log2_group_sz,
    DataT* grad_data, bool zero_out=true)
{
    sparse2dense_nhwc(errs_data, nimgs, img_nrows, img_ncols, img_ngroups,
        log2_group_sz, grad_data, zero_out);
    // uint16_t group_sz = 1 << log2_group_sz;
    // auto ngroups = img_nchan / group_sz;
    // assert(ngroups * group_sz == img_nchan);

    // auto grads = ar::make_view(grad_data, nimgs, img_nrows, img_ncols, ngroups, group_sz);
    // auto errs = ar::make_view(errs_data, nimgs, img_nrows, img_ncols, ngroups);
}

// dense errs from output -> packed grads for input
// like dense2sparse, except we need to read which one neuron in each
// group was active instead of choosing one to be active
template<class DataT>
static const void sparse2dense_nhwc_grad(const DataT* in_packed,
    int nimgs, int img_nrows, int img_ncols, int img_ngroups,
    uint8_t log2_group_sz,
    const DataT* errs_data, DataT* grad_data, bool zero_out=true)
{
    auto group_sz = ((uint16_t)1) << log2_group_sz;
    auto in = ar::make_view(in_packed, nimgs, img_nrows, img_ncols, img_ngroups);
    auto errs = ar::make_view(errs_data, nimgs, img_nrows, img_ncols, img_ngroups, group_sz);
    auto grads = ar::make_view(grad_data, nimgs, img_nrows, img_ncols, img_ngroups);
    if (zero_out) {
        grads.setZero();
    }

    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < img_nrows; i++) {
            for (int j = 0; j < img_ncols; j++) {
                for (int g = 0; g < ngroups; g++) {
                    uint16_t idx;
                    DataT _;
                    unpack_idx_val(in[{n, i, j, g}], &idx, &_);
                    auto errval = errs[{n, i, j, g, idx}];
                    grads[{n, i, j, g}] = errval;
                }
            }
        }
    }
}

#endif // _ksparse_interop_grad_hpp
