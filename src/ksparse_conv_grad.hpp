//
//  ksparse_conv.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _ksparse_conv_hpp
#define _ksparse_conv_hpp

#include <assert.h>

#include "arrayview.hpp"
#include "dtype_traits.hpp"

// NOTE: we're computing both the gradient wrt the filter coeffs and
// wrt to the input simultaneously, because we like to live on the edge
template<class DataT, class CoeffT>
static const void sparse2sparse_conv2d_nhwc_x_ghwc_valid_grad_filt(
    int out_ngroups, int in_log2_group_sz,
    int in_ngroups, int out_log2_group_sz,
    const DataT* in_packed, int nimgs, int img_nrows, int img_ncols,
    const CoeffT* filt_data, int filt_nrows, int filt_ncols,
    // const DataT* errs_packed, DataT* grad_filt)
    const DataT* errs_packed, DataT* grad_filt, DataT* grad_in_packed)
{
    auto nrow_positions = img_nrows - filt_nrows + 1;
    auto ncol_positions = img_ncols - filt_ncols + 1;
    auto out_nrows = nrow_positions; // adjust if strided
    auto out_ncols = ncol_positions; // adjust if strided
    assert(nrow_positions > 0);
    assert(ncol_positions > 0);

    auto in_group_sz = ((uint16_t)1) << in_log2_group_sz;
    auto out_group_sz = ((uint16_t)1) << out_log2_group_sz;

    auto in = ar::make_view(in_packed, nimgs, img_nrows, img_ncols, in_ngroups);
    auto grad_in = ar::make_view(grad_in_packed, nimgs, img_nrows, img_ncols, in_ngroups);
    auto errs = ar::make_view(errs_packed, nimgs, out_nrows, out_ncols, out_ngroups);
    auto filt = ar::make_view(filt_data, out_ngroups, out_group_sz,
        filt_nrows, filt_ncols, in_ngroups, in_group_sz);
    auto grad_f = ar::make_view(filt_data, out_ngroups, out_group_sz,
        filt_nrows, filt_ncols, in_ngroups, in_group_sz);

    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < nrow_positions; i++) {
            for (int j = 0; j < ncol_positions; j++) {
                for (int g = 0; g < out_ngroups; g++) {
                    DataT err;
                    uint16_t gg;
                    unpack_idx_val(errs[{n, i, j, g}], &gg, &err);
                    // gradient wrt filter params
                    // iterate thru all filter params at this position
                    for (int c = 0; c < in_ngroups; c++) {
                        for (int ii = 0; ii < filt_nrows; ii++) {
                            for (int jj = 0; jj < filt_ncols; jj++) {
                                DataT in_act;
                                uint16_t cc;
                                unpack_idx_val(in[{n, i+ii, j+jj, c}],
                                    &cc, &in_act);
                                grad_f[{g, gg, ii, jj, c, cc}] += in_act * err;

                                // gradient wrt input
                                auto coeff = filt[{g, gg, ii, jj, c, cc}];
                                grad_in[{n, i+ii, j+jj, c}] += err * coeff;

                                // in_i = i + filt_nrows - ii - 1;
                                // in_j = j + filt_ncols - jj - 1;
                                // if (in_i >= 0 && in_j >= 0) {
                                //     auto coeff = filt[{g, gg, ii, jj, c, cc}];
                                //     grad_in[{n, in_i, in_j, c}] += err * coeff;
                                // }
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif // _ksparse_conv_hpp
