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
    int ncard, ResultT* out_data)
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

template<class DataT, class CoeffT,
    class ResultT=typename scalar_traits<DataT, CoeffT>::prod_type,
    REQUIRE_INT(DataT)>
void cat2cat_conv2d_nchw_x_gvchw_valid(
    const DataT* img_data, int nimgs, int img_nvars, int img_nrows,
    int img_ncols,
    const CoeffT* filt_data, int nout, int ncard, int filt_nrows, int filt_ncols,
    ResultT* out_data, int nvars_per_out=-1)
{
    if (nvars_per_out < 1) { nvars_per_out = img_nvars; }

    auto nrow_positions = img_nrows - filt_nrows + 1;
    auto ncol_positions = img_ncols - filt_ncols + 1;
    auto out_nrows = nrow_positions; // adjust if strided
    auto out_ncols = ncol_positions; // adjust if strided

    // auto nvars_per_out = img_nvars; // all outputs listen to all inputs

    assert(img_nvars % nvars_per_out == 0);
    assert(img_nvars == nvars_per_out); // TODO rm fully connected constraint
    assert(nimgs > 0);
    assert(img_nvars > 0);
    assert(nout > 0);
    assert(nrow_positions > 0);
    assert(ncol_positions > 0);

    auto in = ar::make_view(img_data, nimgs, img_nvars, img_nrows, img_ncols);
    auto filt = ar::make_view(filt_data, nout, nvars_per_out, ncard, filt_nrows, filt_ncols);
    auto out = ar::make_view(out_data, nimgs, nout, out_nrows, out_ncols);

    out.setZero();

    for (int n = 0; n < nimgs; n++) { // for each img
        for (int g = 0; g < nout; g++) { // for each output channel
            for (int v = 0; v < img_nvars; v++) { // for each input var
                for (int i = 0; i < nrow_positions; i++) { // conv2d
                    for (int j = 0; j < ncol_positions; j++) {
                        for (int k = 0; k < filt_nrows; k++) {
                            for (int l = 0; l < filt_ncols; l++) {
                                auto idx = in[{n, v, i, j}];
                                out[{n, g, i, j}] += filt[{g, v, idx, i+k, j+l}];
                            }
                        }
                    }
                }
            }
        }
    }
}

// do the argmax
template<class ExcitationT, class IdxT>
void argmax_nchw_activations(
    const ExcitationT* exc_data, int nimgs, int nchan, int nrows, int ncols,
    IdxT* out_idxs, int out_nvars, ExcitationT* maxes_tmp)
{
    IdxT group_sz = static_cast<IdxT>(nchan / out_nvars);
    assert(group_sz * out_nvars == nchan); // TODO different group sizes?

    // auto in = ar::make_view(exc_data, nimgs, nchan, nrows, ncols);
    auto in = ar::make_view(exc_data, nimgs, out_nvars, (int)group_sz, nrows, ncols);
    auto out = ar::make_view(out_idxs, nimgs, out_nvars, nrows, ncols);
    auto maxes = ar::make_view(maxes_tmp, nrows, ncols);

    // TODO this will probably work better with nhwc so that we don't need
    // temp array for maxes
    //  -would also make non-orthogonal MCQ easier

    for (int n = 0; n < nimgs; n++) { // each img
        for (int g = 0; g < out_nvars; g++) { // each output channel
            maxes.setValue(std::numeric_limits<ExcitationT>::lowest());
            for (IdxT c = 0; c < group_sz; c++) { // each neuron in group
                for (int i = 0; i < nrows; i++) { // each row
                    for (int j = 0; j < ncols; j++) { // each col
                        auto excitation = in[{n,g,c,i,j}];
                        // use "<=" so out can't be undefined, even if
                        // excitation is min possible value for its type
                        if (maxes[{i, j}] <= excitation) {
                            out[{n,g,i,j}] = c;
                            maxes[{i, j}] = excitation;
                        }
                    }
                }
            }
        }
    }
}

// what I should do is figure out how to slice this such that it returns
// a tensor of lower rank and implement one recursive function; but the
// eigen tensor API sucks, so longer, hackier code it is
template<class EigenTensorT>
void print_tensor5(const EigenTensorT& tensor, const char* name="") {
    if (strlen(name)) { printf("%s:\n", name); }

    auto shape = tensor.dimensions();

    // for (int n = 1; n < 2; n++) {
    for (int n = 0; n < shape[0]; n++) {
        printf("[");
        for (int c = 0; c < shape[1]; c++) {
            if (c > 0) { printf(" "); }
            printf("[");
            for (int i = 0; i < shape[2]; i++) {
                if (i > 0) { printf("\t"); }
                printf("[");
                for (int j = 0; j < shape[3]; j++) {
                    printf("[ ");
                    for (int k = 0; k < shape[4]; k++) {
                        std::cout << tensor(n, c, i, j, k);
                        if (k < shape[4] - 1) { printf(", "); }
                    }
                    printf("]");
                }
                printf("]");
            }
            printf("]");
            if (c < shape[1] - 1) { printf("\n"); }
        }
        printf("]\n");
        if (n < shape[0] - 1) { printf("\n"); }
    }
}


template<class EigenTensorT>
void print_tensor4(const EigenTensorT& tensor, const char* name="") {
    if (strlen(name)) { printf("%s:\n", name); }

    auto shape = tensor.dimensions();

    // for (int n = 1; n < 2; n++) {
    for (int n = 0; n < shape[0]; n++) {
        printf("[");
        for (int c = 0; c < shape[1]; c++) {
            if (c > 0) { printf(" "); }
            printf("[");
            for (int i = 0; i < shape[2]; i++) {
                printf("[ ");
                for (int j = 0; j < shape[3]; j++) {
                    std::cout << tensor(n, c, i, j);
                    if (j < shape[3] - 1) { printf(", "); }
                }
                printf("]");
            }
            printf("]");
            if (c < shape[1] - 1) { printf("\n"); }
        }
        printf("]\n");
        if (n < shape[0] - 1) { printf("\n"); }
    }
}

#endif // _catconv_hpp
