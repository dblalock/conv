//
//  ksparse_conv_grad.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _ksparse_conv_grad_hpp
#define _ksparse_conv_grad_hpp

#include <assert.h>
// #include <type_traits>

#include "arrayview.hpp"
#include "dtype_traits.hpp"
#include "ksparse_util.hpp"

// static const int frobnicate() { return 7; }

// NOTE: we're computing both the gradient wrt the filter coeffs and
// wrt to the input simultaneously, because we like to live on the edge
template<class DataT, class ParamT>
static const void sparse2sparse_conv2d_nhwc_x_ghwc_valid_grad(
    const DataT* in_packed, int nimgs, int img_nrows, int img_ncols,
    const ParamT* filt_data, int out_ngroups, int out_log2_group_sz,
    int filt_nrows, int filt_ncols, int in_ngroups, int in_log2_group_sz,
    // const DataT* errs_packed, DataT* grad_filt)
    const DataT* errs_packed, DataT* grad_filt, DataT* grad_in_packed,
    bool zero_grad_filt=true, bool zero_grad_in=true)
{
    // static_assert(!std::is_const<DataT>::value, "DataT can't be a const type!");
    const int in_nbits = in_log2_group_sz;
    const int out_nbits = out_log2_group_sz;
    auto nrow_positions = img_nrows - filt_nrows + 1;
    auto ncol_positions = img_ncols - filt_ncols + 1;
    auto out_nrows = nrow_positions; // adjust if strided
    auto out_ncols = ncol_positions; // adjust if strided
    assert(nrow_positions > 0);
    assert(ncol_positions > 0);

    auto in_group_sz = ((uint16_t)1) << in_log2_group_sz;
    auto out_group_sz = ((uint16_t)1) << out_log2_group_sz;

    auto in = ar::make_view(
        in_packed, nimgs, img_nrows, img_ncols, in_ngroups);
    auto grad_in = ar::make_view(
        grad_in_packed, nimgs, img_nrows, img_ncols, in_ngroups);
    auto errs = ar::make_view(
        errs_packed, nimgs, out_nrows, out_ncols, out_ngroups);
    auto filt = ar::make_view(filt_data, out_ngroups, out_group_sz,
        filt_nrows, filt_ncols, in_ngroups, in_group_sz);
    auto grad_f = ar::make_view(grad_filt, out_ngroups, out_group_sz,
        filt_nrows, filt_ncols, in_ngroups, in_group_sz);
    if (zero_grad_filt) { grad_f.setZero(); }
    if (zero_grad_in) { grad_in.setZero(); }

    // the idea here is we basically do the same thing as the forward operation,
    // except we accumulate gradients instead of activations; this approach
    // only works since this impl is single-threaded; otherwise we'd end up
    // with races when trying to add to grads for filter params, since every
    // position can update any of the params
    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < out_nrows; i++) {
            for (int j = 0; j < out_ncols; j++) {
                for (int g = 0; g < out_ngroups; g++) {
                    DataT err;
                    uint16_t gg;
                    unpack_idx_val(errs[{n, i, j, g}], out_nbits, &gg, &err);

                    // if (err != 648) { printf("WTF (%d,%d,%d,%d) err = %g\n", n, i, j, g, err); }
                    // gradient wrt filter params
                    // iterate thru all filter params at this position
                    for (int ii = 0; ii < filt_nrows; ii++) {
                        for (int jj = 0; jj < filt_ncols; jj++) {
                            for (int c = 0; c < in_ngroups; c++) {
                                DataT in_act;
                                uint16_t cc;
                                unpack_idx_val(in[{n, i+ii, j+jj, c}],
                                    in_nbits, &cc, &in_act);
                                grad_f[{g, gg, ii, jj, c, cc}] += in_act * err;

                                // so why is cc not always in_gruop_sz - 1?

                                // if (true) {
                                //     printf("at output=(%d,%d,%d,%d), ", n, i, j, g);
                                //     printf("input=(%d,%d,%d,%d), ", n, i + ii, j+jj, c);
                                //     printf("grad_f[%d,%d,%d,%d,%d,%d] ", g, gg, ii, jj, c, cc);
                                //     printf("+= %.7g*%.7g=%.7g; grad_f=%.7g\n", in_act, err, in_act * err, grad_f[{g, gg, ii, jj, c, cc}]);
                                // }

                                // gradient wrt input
                                auto coeff = filt[{g, gg, ii, jj, c, cc}];
                                grad_in[{n, i+ii, j+jj, c}] += err * coeff;

                                // if (coeff != 2) { printf("WTF (%d,%d,%d,%d) coeff = %g\n", n, i, j, g, coeff); }
                                // if (err * coeff != 1296) { printf("WTF (%d,%d,%d,%d) adding to grad: %.7g\n", n, i, j, c, err * coeff); }
                                // if (true) { printf("at (%d,%d,%d,%d) adding to grad: %.7g; new total grad: %.7g\n", n, i+ii, j+jj, c, err * coeff, grad_in[{n, i+ii, j+jj, c}]); }

                                // in_i = i + filt_nrows - ii - 1;
                                // in_j = j + filt_ncols - jj - 1;
                                // if (in_i >= 0 && in_j >= 0) {
                                //     auto coeff = filt[{g, gg, ii, jj, c, cc}];
                                //     grad_in[{n, in_i, in_j, c}] += err * coeff;
                                // }
                            }
                        }
                        // auto g_in = grad_in[{n, i, j, c}];
                        // if (g_in != 1296) { printf("WTF (%d,%d,%d,%d) x_grad = %g\n", n, i, j, c, g_in); }
                    }
                }
            }
        }
        // pack input indices into raw input gradients; this way the next layer
        // knows what its output indices were
        // TODO can we just get the output to pass into the gradient op?
        for (int i = 0; i < img_nrows; i++) {
            for (int j = 0; j < img_ncols; j++) {
                for (int c = 0; c < in_ngroups; c++) {
                    DataT _;
                    uint16_t cc;

                    // TODO rm
                    // auto g_in = grad_in[{n, i, j, c}];
                    // if (g_in != 1296) { printf("WTF (%d,%d,%d,%d) x_grad = %g\n", n, i, j, c, g_in); }

                    unpack_idx_val(in[{n, i, j, c}], in_nbits, &cc, &_);
                    grad_in[{n, i, j, c}] = pack_idx_val(
                        in_nbits, cc, grad_in[{n, i, j, c}]);
                }
            }
        }
    }
}

#endif // _ksparse_conv_grad_hpp
