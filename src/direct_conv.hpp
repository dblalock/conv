//
//  direct_conv.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _direct_conv_hpp
#define _direct_conv_hpp

#include <assert.h>
#include <stdio.h>

#include "arrayview.hpp"
#include "dtype_traits.hpp"


// only allow valid convolution
template<class DataT, class CoeffT,
    class ResultT=typename scalar_traits<DataT, CoeffT>::prod_type>
void conv2dx2d_valid_rowmajor(
    const DataT* img_data, int img_nrows, int img_ncols,
    const CoeffT* filt_data, int filt_nrows, int filt_ncols,
    CoeffT* out_data)
{
    auto nrow_positions = img_nrows - filt_nrows + 1;
    auto ncol_positions = img_ncols - filt_ncols + 1;
    auto out_nrows = nrow_positions; // adjust if strided
    auto out_ncols = ncol_positions; // adjust if strided

    assert(nrow_positions > 0);
    assert(ncol_positions > 0);

    auto in = ar::make_view(img_data, img_nrows, img_ncols);
    auto filt = ar::make_view(filt_data, filt_nrows, filt_ncols);
    auto out = ar::make_view(out_data, out_nrows, out_ncols);

    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            out[{i, j}] = 0;
            for (int k = 0; k < filt_nrows; k++) {
                for (int l = 0; l < filt_ncols; l++) {
                    out[{i, j}] += in[{i + k, j + l}] * filt[{k, l}];
                }
            }
        }
    }
}

// template<bool ChannelsFirst, class DataT, class CoeffT,
template<class DataT, class CoeffT,
    class ResultT=typename scalar_traits<DataT, CoeffT>::prod_type>
void conv3dx3d_valid_hwc(
    const DataT* img_data, int img_nrows, int img_ncols,
    const CoeffT* filt_data, int filt_nrows, int filt_ncols,
    int nchannels, CoeffT* out_data)
{
    auto nrow_positions = img_nrows - filt_nrows + 1;
    auto ncol_positions = img_ncols - filt_ncols + 1;
    // auto nchan_positions = 1;
    auto out_nrows = nrow_positions; // adjust if strided
    auto out_ncols = ncol_positions; // adjust if strided

    assert(nrow_positions > 0);
    assert(ncol_positions > 0);

    auto in = ar::make_view(img_data, img_nrows, img_ncols, nchannels);
    auto filt = ar::make_view(filt_data, filt_nrows, filt_ncols, nchannels);
    auto out = ar::make_view(out_data, out_nrows, out_ncols);

    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            out[{i, j}] = 0;
            for (int k = 0; k < filt_nrows; k++) {
                for (int l = 0; l < filt_ncols; l++) {
                    for (int c = 0; c < nchannels; c++) {
                        out[{i, j}] += in[{i + k, j + l, c}] * filt[{k, l, c}];
                    }
                }
            }
        }
    }
}

template<class DataT, class CoeffT,
    class ResultT=typename scalar_traits<DataT, CoeffT>::prod_type>
void conv3dx3d_valid_chw(
    const DataT* img_data, int img_nrows, int img_ncols,
    const CoeffT* filt_data, int filt_nrows, int filt_ncols,
    int nchannels, CoeffT* out_data)
{
    auto nrow_positions = img_nrows - filt_nrows + 1;
    auto ncol_positions = img_ncols - filt_ncols + 1;
    // auto nchan_positions = 1;
    auto out_nrows = nrow_positions; // adjust if strided
    auto out_ncols = ncol_positions; // adjust if strided

    assert(nrow_positions > 0);
    assert(ncol_positions > 0);

    auto in = ar::make_view(img_data, nchannels, img_nrows, img_ncols);
    auto filt = ar::make_view(filt_data, nchannels, filt_nrows, filt_ncols);
    auto out = ar::make_view(out_data, out_nrows, out_ncols);

    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            out[{i, j}] = 0; // zero whole output
        }
    }
    for (int c = 0; c < nchannels; c++) {
        for (int i = 0; i < nrow_positions; i++) {
            for (int j = 0; j < ncol_positions; j++) {
                // printf("X(%d,%d,%d) = %g\n", c, i, j, in[{c, i, j}]);
                for (int k = 0; k < filt_nrows; k++) {
                    for (int l = 0; l < filt_ncols; l++) {
                        out[{i, j}] += in[{c, i + k, j + l}] * filt[{c, k, l}];
                    }
                }
            }
        }
    }
}

#endif // _direct_conv_hpp
