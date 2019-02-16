//
//  test_catconv.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "../src/catconv.hpp"

#include <gtest/gtest.h>
#include "testing_utils.hpp"

TEST(CatConv, SmokeTest) {
    auto x = 5;
    EXPECT_EQ(x, 5);
    EXPECT_GT(x, 4);
}

TEST(CatConv, 2d_hw_chw_valid) {
    auto ncard = 4;
    auto nrows = 5;
    auto ncols = 3;
    Ar2D<int> X(nrows, ncols);
    auto filt_nrows = 2;
    auto filt_ncols = 2;
    Ar3D<float> filt(ncard, filt_nrows, filt_ncols);
    auto nrow_positions = nrows - filt_nrows + 1;
    auto ncol_positions = ncols - filt_ncols + 1;
    Ar2D<float> out(nrow_positions, ncol_positions);
    Ar2D<float> ans(nrow_positions, ncol_positions);
    for (int i = 0; i < X.size(); i++) { X.data()[i] = i % ncard; }
    for (int i = 0; i < filt.size(); i++) {
        static float vals[8] = {1,0,-1,2, -2,3,4,-3};
        filt.data()[i] = vals[i % 8];
    }
    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            ans(i, j) = filt(X(i,j),0,0) + filt(X(0+i,1+j),0,1) +
                filt(X(1+i,0+j),1,0) + filt(X(1+i,1+j),1,1);
            // printf("ans(%d,%d) = %g\n", i, j, ans(i, j));
        }
    }

    catconv2d_hw_x_chw_valid(X.data(), nrows, ncols,
        filt.data(), filt_nrows, filt_ncols, ncard, out.data());

    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            EXPECT_EQ(out(i, j), ans(i, j)) << "i, j = " << i << ", " << j;
        }
    }
}

TEST(CatConv, 2d_nchw_x_gvchw_valid) {
    auto nimgs = 6;
    auto nvars = 4;
    auto nrows = 5;
    auto ncols = 3;
    Ar4D<int> X(nimgs, nvars, nrows, ncols);
    auto nout = 6;
    auto nvars_per_out = nvars; // TODO make this nvars / nout
    auto ncard = 7;  // yields variable values in output
    auto filt_nrows = 2;
    auto filt_ncols = 2;
    Ar5D<float> filt(nout, nvars_per_out, ncard, filt_nrows, filt_ncols);
    auto nrow_positions = nrows - filt_nrows + 1;
    auto ncol_positions = ncols - filt_ncols + 1;
    Ar4D<float> out(nimgs, nout, nrow_positions, ncol_positions);
    Ar4D<float> ans(nimgs, nout, nrow_positions, ncol_positions);
    for (int i = 0; i < X.size(); i++) { X.data()[i] = i % ncard; }
    for (int i = 0; i < filt.size(); i++) {
        static float vals[8] = {1,0,-1,2, -2,3,4,-3};
        filt.data()[i] = vals[i % 8];
    }
    // for (int i = 0; i < nrow_positions; i++) {
    //     for (int j = 0; j < ncol_positions; j++) {
    //         ans(i, j) = filt(X(i,j),0,0) + filt(X(0+i,1+j),0,1) +
    //             filt(X(1+i,0+j),1,0) + filt(X(1+i,1+j),1,1);
    //         // printf("ans(%d,%d) = %g\n", i, j, ans(i, j));
    //     }
    // }

    // XXX we don't actually verify correctness; just compilation
    cat2cat_conv2d_nchw_x_gvchw_valid(X.data(), nimgs, nvars, nrows, ncols,
        filt.data(), nout, ncard, filt_nrows, filt_ncols, out.data());

    auto ngroups = nout / 2;
    auto group_sz = nout / ngroups;

    Ar2D<float> tmp(nrow_positions, ncol_positions);
    Ar4D<uint8_t> argmaxes(nimgs, ngroups, nrow_positions, ncol_positions);
    argmax_nchw_activations(
        out.data(), nimgs, nout, nrow_positions, ncol_positions,
        argmaxes.data(), ngroups, tmp.data());

    Eigen::TensorMap<Ar5D<float> > excitations = Eigen::TensorMap<Ar5D<float> >(
        out.data(), nimgs, ngroups, group_sz, nrow_positions, ncol_positions);

    // print_tensor5(excitations, "Excitations");
    // Ar4D<int> argmaxes_print = argmaxes.cast<int>();
    // print_tensor4(argmaxes_print, "Argmaxes");

    // check that everything the argmax should have compared is <= the
    // value at the reported argmax
    // for (int n = 1; n < 2; n++) {
    for (int n = 0; n < nimgs; n++) {
        for (int g = 0; g < ngroups; g++) {
            // auto group_start_channel = g * group_sz;
            for (int c = 0; c < group_sz; c++) {
                // auto channel_idx = g * group_sz + c;
                for (int i = 0; i < nrow_positions; i++) {
                    for (int j = 0; j < ncol_positions; j++) {
                        int idx = argmaxes(n, g, i, j);
                        // auto max_channel = group_start_channel + idx;
                        // auto val = out(n, channel_idx, i, j);
                        // auto max = out(n, max_channel, i, j);
                        auto max = excitations(n, g, idx, i, j);
                        auto val = excitations(n, g, c, i, j);
                        // auto max = MAX(excitations(n, g, 0, i, j), excitations(n, g, 1, i, j));
                        EXPECT_GE(max, val) << "argmax " << idx <<
                            " wrong at (n, g, c, i, j):\t("
                            << n << ", " << g << ", " << c << ", " <<
                            i << ", " << j << ")\n";
                    }
                }
            }
        }
    }

    // for (int i = 0; i < nrow_positions; i++) {
    //     for (int j = 0; j < ncol_positions; j++) {
    //         EXPECT_EQ(out(i, j), ans(i, j)) << "i, j = " << i << ", " << j;
    //     }
    // }
}
