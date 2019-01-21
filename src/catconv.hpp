//
//  catconv.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _catconv_hpp
#define _catconv_hpp

#include <assert.h>
#include <stdio.h>

#include "arrayview.hpp"
#include "dtype_traits.hpp"




template<class DataT, class CoeffT,
    class ResultT=typename scalar_traits<DataT, CoeffT>::prod_type,
    REQUIRE_INT(DataT)>
void catconv2d_hw_x_chw_valid(
    const DataT* img_data, int img_nrows, int img_ncols,
    const CoeffT* filt_data, int filt_nrows, int filt_ncols,
    int ncard, CoeffT* out_data)
{
    auto nrow_positions = img_nrows - filt_nrows + 1;
    auto ncol_positions = img_ncols - filt_ncols + 1;
    auto out_nrows = nrow_positions; // adjust if strided
    auto out_ncols = ncol_positions; // adjust if strided

    assert(nrow_positions > 0);
    assert(ncol_positions > 0);

    auto in = ar::make_view(img_data, img_nrows, img_ncols);
    auto filt = ar::make_view(filt_data, ncard, filt_nrows, filt_ncols);
    auto out = ar::make_view(out_data, out_nrows, out_ncols);

    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            out[{i, j}] = 0;
            for (int k = 0; k < filt_nrows; k++) {
                for (int l = 0; l < filt_ncols; l++) {
                    auto idx = in[{i + k, j + l}];
                    out[{i, j}] +=  filt[{idx, k, l}];
                }
            }
        }
    }
}



#endif // _catconv_hpp
