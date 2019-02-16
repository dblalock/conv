//
//  ksparse_interop.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _ksparse_interop_hpp
#define _ksparse_interop_hpp

#include <assert.h>

#include "arrayview.hpp"
#include "ksparse_util.hpp"


// template<class DataT, class IdxT=uint8_t>
// static const void sparse2dense_nhwc_nopacking(
//     const DataT* in_activations, int nimgs, int img_nrows, int img_ncols,
//     int img_ngroups, uint8_t log2_group_sz,
//     const IdxT* in_idxs, DataT* out_data)
// {
//     auto group_sz = 1 << log2_group_sz;
//     // nout = group_sz * ngroups;

//     auto in_act = ar::make_view(in_activations, nimgs, img_nrows, img_ncols, img_ngroups);
//     auto in_idxs = ar::make_view(in_idxs, nimgs, img_nrows, img_ncols, img_ngroups);
//     auto out = ar::make_view(out_data, nimgs, img_nrows, img_ncols, img_ngroups, group_sz);
//     out.setZero();

//     for (int n = 0; n < nimgs; n++) {
//         for (int i = 0; i < nrow_positions; i++) {
//             for (int j = 0; j < ncol_positions; j++) {
//                 for (int g = 0; g < ngroups; g++) {
//                     auto idx = in_idxs[{n, i, j, g}];
//                     out[{n, i, j, g, idx}] = in_act[{n, i, j, g}];
//                 }
//             }
//         }
//     }
// }



// template<class DataT=void>
template<int Log2GroupSz=-1, class DataT=void>
static const void dense2sparse_nhwc(const DataT* img_data, int nimgs,
    int img_nrows, int img_ncols, int img_nchan,
    DataT* out_data, uint8_t log2_group_sz=0)
{
    // static_assert(Log2GroupSz <= 15, "Max Group Size is 1 << 15");
    const int nbits = Log2GroupSz > 0 ? Log2GroupSz : log2_group_sz;
    assert(nbits > 0); // must specify group sz either statically or dynamically
    const int group_sz = 1 << nbits;
    const int ngroups = img_nchan / group_sz;
    assert(ngroups * group_sz == img_nchan); // TODO allow unequel group sizes

    auto in = ar::make_view(img_data, nimgs, img_nrows, img_ncols, ngroups, group_sz);
    auto out = ar::make_view(out_data, nimgs, img_nrows, img_ncols, ngroups);

    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < img_nrows; i++) {
            for (int j = 0; j < img_ncols; j++) {
                // compute max activation in each group
                for (int g = 0; g < ngroups; g++) {
                    // to avoid branching on val > maxval, we just always
                    // pack indices into the low bits of the vals and take
                    // the max of the packed values; this is correct given
                    // that low bits will get thrown away either
                    // way (although not 100% certain it's true for floats)
                    auto maxval = in[{n, i, j, g, 0}];
                    // printf("got first maxval: %g\n", (float)maxval);
                    DataT packed_max = pack_idx_val(nbits, (uint8_t)0, maxval);
                    // printf("first packed_max: %g\n", (float)packed_max);
                    // #pragma unroll
                    for (uint16_t gg = 1; gg < group_sz; gg++) {
                        auto val = in[{n, i, j, g, gg}];
                        auto packed_val = pack_idx_val(nbits, gg, val);
                        packed_max = MAX(packed_max, packed_val);
                    }
                    // printf("writing out packed_max: %g\n", (float)packed_max);
                    out[{n, i, j, g}] = packed_max;
                }
            }
        }
    }
}
// template<class DataT>
template<int Log2GroupSz=-1, class DataT=void>
static const void sparse2dense_nhwc(
    const DataT* in_packed,
    int nimgs, int img_nrows, int img_ncols, int img_ngroups,
    DataT* out_data, uint8_t log2_group_sz=-1, bool zero_out=true)
{
    const int nbits = Log2GroupSz > 0 ? Log2GroupSz : log2_group_sz;
    assert(nbits > 0); // must specify group sz either statically or dynamically
    auto group_sz = ((uint16_t)1) << nbits;
    auto in = ar::make_view(in_packed, nimgs, img_nrows, img_ncols, img_ngroups);
    auto out = ar::make_view(out_data, nimgs, img_nrows, img_ncols, img_ngroups, group_sz);
    if (zero_out) {
        out.setZero();
    }
    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < img_nrows; i++) {
            for (int j = 0; j < img_ncols; j++) {
                for (int g = 0; g < img_ngroups; g++) {
                    uint16_t idx;
                    DataT val;
                    unpack_idx_val(in[{n, i, j, g}], nbits, &idx, &val);
                    // printf("unpacked to idx, val: %u, %g\n", idx, (float)val);
                    out[{n, i, j, g, idx}] = val;
                }
            }
        }
    }
}


#endif // _ksparse_interop_hpp
