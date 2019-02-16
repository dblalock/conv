//
//  ksparse_conv.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _ksparse_conv_hpp
#define _ksparse_conv_hpp

#include <assert.h>

#include "arrayview.hpp"
#include "dtype_traits.hpp"
#include "macros.hpp"  // for type traits macros

// template<class DataT, class CoeffT,
//     class ActivationT=typename scalar_traits<DataT, CoeffT>::prod_type,
//     class IdxT=uint8_t>
// static const void dense2sparse_conv2d_nhwc_x_ghwc_valid(
//     const DataT* img_data, int nimgs, int img_nrows, int img_ncols, int img_nchan,
//     const CoeffT* filt_data, int nout, int filt_nrows, int filt_ncols,
//     ActivationT* out_activations, IdxT* out_indices, uint8_t log2_group_sz=2)
// {
//     auto nrow_positions = img_nrows - filt_nrows + 1;
//     auto ncol_positions = img_ncols - filt_ncols + 1;
//     auto out_nrows = nrow_positions; // adjust if strided
//     auto out_ncols = ncol_positions; // adjust if strided
//     assert(nrow_positions > 0);
//     assert(ncol_positions > 0);

//     auto group_sz = 1 << log2_group_sz;
//     auto ngroups = nout / group_sz;
//     assert(ngroups * group_sz == nout); // TODO allow unequally sized groups

//     auto in = ar::make_view(img_data, nimgs, img_nrows, img_ncols, img_nchan);
//     auto filt = ar::make_view(filt_data, ngroups, group_sz, filt_nrows, filt_ncols, img_nchan);
//     auto out_act = ar::make_view(out_activations, nimgs, out_nrows, out_ncols, ngroups);
//     auto out_idxs = ar::make_view(out_indices, nimgs, out_nrows, out_ncols, ngroups);

//     auto min_possible_activation = std::numeric_limits<ActivationT>::lowest();

//     for (int n = 0; n < nimgs; n++) {
//         for (int i = 0; i < nrow_positions; i++) {
//             for (int j = 0; j < ncol_positions; j++) {
//                 // compute max activation in each group
//                 for (int g = 0; g < ngroups; g++) {
//                     auto max_act = min_possible_activation;
//                     auto argmax_idx = 0;
//                     for (int gg = 0; gg < group_sz; gg++) {
//                         // compute activation for this neuron in group
//                         ActivationT act = 0;
//                         for (int ii = 0; ii < filt_nrows; ii++) {
//                             for (int jj = 0; jj < filt_ncols; jj++) {
//                                 for (int c = 0; c < img_nchan; c++) {
//                                     act += in[{n, i+ii, j+jj, c}] * filt[{g, gg, ii, jj, c}]
//                                 }
//                             }
//                         }
//                         // update best-so-far (activation, idx) pair
//                         if (act > max_act) {
//                             max_act = act;
//                             argmax_idx = gg;
//                         }
//                     }
//                 }
//                 // out_act[{n, i, j, g}] = MAX(0, max_act); // relu
//                 out_act[{n, i, j, g}] = max_act;
//                 out_idxs[{n, i, j, g}] = argmax_idx;
//             }
//         }
//     }
// }

template<class DataT, class CoeffT>
static const void sparse2sparse_conv2d_nhwc_x_ghwc_valid(
    int out_ngroups, int in_log2_group_sz,
    int in_ngroups, int out_log2_group_sz,
    const DataT* in_packed, int nimgs, int img_nrows, int img_ncols,
    const CoeffT* filt_data, int filt_nrows, int filt_ncols,
    DataT* out_packed)
{
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

    auto in = ar::make_view(in_packed, nimgs, img_nrows, img_ncols, in_ngroups);
    auto out = ar::make_view(out_packed, nimgs, out_nrows, out_ncols, out_ngroups);
    auto filt = ar::make_view(filt_data, out_ngroups, out_group_sz,
        filt_nrows, filt_ncols, in_ngroups, in_group_sz);

    auto min_possible_activation = std::numeric_limits<DataT>::lowest();

    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < nrow_positions; i++) {
            for (int j = 0; j < ncol_positions; j++) {
                // compute max activation in each group
                for (int g = 0; g < out_ngroups; g++) {
                    auto max_act = min_possible_activation;
                    for (uint16_t gg = 0; gg < out_group_sz; gg++) {
                        // compute activation for one output neuron in group
                        DataT act = 0;
                        for (int ii = 0; ii < filt_nrows; ii++) {
                            for (int jj = 0; jj < filt_ncols; jj++) {
                                #pragma unroll
                                for (int c = 0; c < in_ngroups; c++) {
                                    DataT in_act;
                                    uint16_t cc;
                                    unpack_idx_val(in[{n, i+ii, j+jj, c}],
                                        in_nbits, &cc, &in_act);
                                    auto coeff = filt[{g, gg, ii, jj, c, cc}];
                                    act += in_act * coeff;
                                }
                            }
                        }
                        // compare this neuron to max activation so far; we
                        // do max on packed repr to avoid conditional branch
                        act = pack_idx_val(out_nbits, gg, act);
                        max_act = MAX(max_act, act);
                    }
                    out[{n, i, j, g}] = max_act;
                }
            }
        }
    }
}

// template<int InLog2GroupSz, int OutLog2GroupSz,
//     class DataT, class CoeffT, class IdxT>
// static const void sparse2sparse_conv2d_nhwc_x_ghwc_valid_nopacking(
//     int in_ngroups, int out_ngroups,
//     const DataT* in_activations, int nimgs, int img_nrows, int img_ncols,
//     const IdxT* in_indices,
//     const CoeffT* filt_data, int filt_nrows, int filt_ncols,
//     DataT* out_activations, IdxT* out_indices)
// {
//     auto nrow_positions = img_nrows - filt_nrows + 1;
//     auto ncol_positions = img_ncols - filt_ncols + 1;
//     auto out_nrows = nrow_positions; // adjust if strided
//     auto out_ncols = ncol_positions; // adjust if strided
//     assert(nrow_positions > 0);
//     assert(ncol_positions > 0);

//     auto in_group_sz = 1 << InLog2GroupSz;
//     auto out_group_sz = 1 << OutLog2GroupSz;
//     auto nout = out_group_sz * out_ngroups; // TODO only cuz arrayview can't do rank 6 tensors

//     auto in_act = ar::make_view(in_activations, nimgs, img_nrows, img_ncols, in_ngroups);
//     auto in_idxs = ar::make_view(in_indices, nimgs, img_nrows, img_ncols, in_ngroups);
//     auto out_act = ar::make_view(out_activations, nimgs, out_nrows, out_ncols, out_ngroups);
//     auto out_idxs = ar::make_view(out_indices, nimgs, out_nrows, out_ncols, out_ngroups);
//     auto filt = ar::make_view(filt_data, out_ngroups, out_group_sz, filt_nrows, filt_ncols, in_ngroups, in_group_sz);

//     auto min_possible_activation = std::numeric_limits<DataT>::lowest();

//     for (int n = 0; n < nimgs; n++) {
//         for (int i = 0; i < nrow_positions; i++) {
//             for (int j = 0; j < ncol_positions; j++) {
//                 // compute max activation in each group
//                 for (int g = 0; g < out_ngroups; g++) {
//                     auto max_act = min_possible_activation;
//                     auto argmax_idx = 0;
//                     for (int gg = 0; gg < out_group_sz; gg++) {
//                         // compute activation for this neuron in group
//                         DataT act = 0;
//                         for (int ii = 0; ii < filt_nrows; ii++) {
//                             for (int jj = 0; jj < filt_ncols; jj++) {
//                                 for (int c = 0; c < in_ngroups; c++) {
//                                     auto x = in_act[{n, i+ii, j+jj, c}];
//                                     auto cc = in_idxs[{n, i+ii, j+jj, c}];
//                                     auto coeff = filt[{g, gg, ii, jj, c, cc}];
//                                     act += x * coeff;
//                                 }
//                             }
//                         }
//                         // update best-so-far (activation, idx) pair
//                         if (act > max_act) {
//                             max_act = act;
//                             argmax_idx = gg;
//                         }
//                     }
//                 }
//                 out_act[{n, i, j, g}] = max_act;
//                 out_idxs[{n, i, j, g}] = argmax_idx;
//             }
//         }
//     }
// }



#endif // _ksparse_conv_hpp
