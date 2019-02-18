//
//  test_ksparse_interop.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "../src/ksparse_conv.hpp"
#include "../src/ksparse_conv_grad.hpp"

#include <gtest/gtest.h>
#include "testing_utils.hpp"
#include "../src/ksparse_interop.hpp"
#include "../src/ksparse_interop_grad.hpp"

// TODO rm dup code from ksparse_interop test
template<class DataT>
void populate_x(Ar5D<DataT>& X, DataT maxval) {
    // argmax in each group should be at idx (n + i + j) % group_sz
    auto shape = X.dimensions();
    auto nimgs = shape[0];
    auto nrows = shape[1];
    auto ncols = shape[2];
    auto ngroups = shape[3];
    auto group_sz = shape[4];
    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                for (int c = 0; c < ngroups; c++) {
                    auto argmax = (n + i + j) % group_sz;
                    for (int cc = 0; cc < group_sz; cc++) {
                        // if not maxval, just some random-ish numbers that
                        // might elicit incorrect output if code is incorrect
                        X(n, i, j, c, cc) = cc == argmax ? maxval : 9*cc % (int)maxval;
                    }
                }
            }
        }
    }
}


TEST(KSparseConv, Forward) {
    const int in_log2_group_sz = 2;
    const int in_group_sz = 1 << in_log2_group_sz;
    uint16_t in_ngroups = 1;
    auto in_nchan = in_ngroups * in_group_sz;
    // using dtype = int16_t;
    using dtype = float;

    auto nimgs = 6;
    auto in_nrows = 5;
    auto in_ncols = 4;
    // auto nimgs = 1;
    // auto in_nrows = 3;
    // auto in_ncols = 3;

    auto filt_nrows = 3;
    auto filt_ncols = 3;
    auto out_ngroups = 2;
    const uint16_t out_log2_group_sz = 2;
    uint16_t out_group_sz = 1 << out_log2_group_sz;
    auto out_nrows = in_nrows - filt_nrows + 1;
    auto out_ncols = in_ncols - filt_ncols + 1;

    Ar5D<dtype> X(nimgs, in_nrows, in_ncols, in_ngroups, in_group_sz);
    Ar4D<dtype> X_packed(nimgs, in_nrows, in_ncols, in_ngroups);
    Ar4D<dtype> Y_packed(nimgs, out_nrows, out_ncols, out_ngroups);
    Ar5D<dtype> Y_hat(nimgs, out_nrows, out_ncols, out_ngroups, out_group_sz);

    Ar6D<dtype> filt(out_ngroups, out_group_sz, filt_nrows, filt_ncols, in_ngroups, in_group_sz);
    filt.setConstant(1);

    // for (int g = 0; g < filt.shape[0]; g++) {
    //     for (int gg = 0; gg < filt.shape[1]; gg++) {
    //         for (int ii = 0; ii < filt_nrows; ii++) {
    //             for (int jj = 0; jj < filt_ncols; jj++) {
    //                 for (int c = 0; c < in_ngroups; c++) {
    //                     for (int cc = 0; cc < in_group_sz; cc++) {
    //                         filt(g, gg, ii, jj, c, cc) = 1;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // create packed input for conv
    dtype maxval = 9 << MAX(out_log2_group_sz, in_log2_group_sz);
    populate_x(X, maxval);
    dense2sparse_nhwc<in_log2_group_sz>(X.data(), nimgs, in_nrows,
        in_ncols, in_nchan, X_packed.data());

    sparse2sparse_conv2d_nhwc_x_ghwc_valid(
        X_packed.data(), nimgs, in_nrows, in_ncols,
        filt.data(), out_ngroups, out_log2_group_sz, filt_nrows, filt_ncols,
        in_ngroups, in_log2_group_sz, Y_packed.data());

    // convert conv output to dense so we can validate it more easily
    sparse2dense_nhwc(Y_packed.data(), nimgs, out_nrows, out_ncols,
        out_ngroups, out_log2_group_sz, Y_hat.data());
    // sparse2dense_nhwc<log2_group_sz>(X_packed.data(), nimgs, nrows, ncols, ngroups, X_hat.data());


    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < out_nrows; i++) {
            for (int j = 0; j < out_ncols; j++) {
                for (int c = 0; c < out_ngroups; c++) {
                    // auto argmax = (n + i + j) % group_sz;
                    for (int cc = 0; cc < out_group_sz; cc++) {
                        auto target = filt_nrows * filt_ncols * in_ngroups * maxval;
                        target = (cc == out_group_sz - 1) ? target : 0;
                        auto outval = Y_hat(n, i, j, c, cc);
                        // EXPECT_TRUE(outval == target || outval == 0)
                        // bool passed = outval == target || outval == 0;
                        EXPECT_EQ(outval, target)
                        // std::cout << "---------------- " << (passed ? "Passed: " : "FAILED\n")
                            << "cc=" << cc << " target=" << target <<
                            " outval=" << outval <<
                            " packed_val=" << Y_packed(n, i, j, c) <<
                            " n,i,j,c,cc=(" << n << "," << i << "," << j << ","
                            << c << "," << cc << ")\n";
                        // EXPECT_TRUE(passed);
                    }
                }
            }
        }
    }
}
