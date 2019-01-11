
#include <assert.h>
#include <utility>
// #include <iostream>
// #include <memory>
#include <stdio.h>

#include <gtest/gtest.h>

#include "arrayview.hpp"

// #define CATCH_CONFIG_RUNNER
// #include "catch.hpp"

template<class T1, class T2>
struct scalar_traits {
    using sum_type = decltype(std::declval<T1>() + std::declval<T2>());
    using prod_type = decltype(std::declval<T1>() * std::declval<T2>());
};

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
        for (int j = 0; j < nrow_positions; j++) {
            out[{i, j}] = 0;
            for (int k = 0; k < nrow_positions; k++) {
                for (int l = 0; l < nrow_positions; l++) {
                    out[{i, j}] += in[{i + k, j + l}] * filt[{k, l}];
                }
            }
        }
    }
}

