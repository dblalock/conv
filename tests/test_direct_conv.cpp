
#include "../src/direct_conv.hpp"

#include <gtest/gtest.h>
#include "testing_utils.hpp"


TEST(DirectConv, DidItWork) {
    auto x = 5;
    EXPECT_EQ(x, 5);
    EXPECT_GT(x, 4);
}

TEST(Sanity, EigenTensor) {
    Eigen::Tensor<float, 2> X(2, 3);
    EXPECT_EQ(X.size(), 6);
    EXPECT_EQ(X.dimensions()[0], 2);
    EXPECT_EQ(X.dimensions()[1], 3);
}


TEST(DirectConv, 2dx2d_rowmajor) {
    auto nrows = 2;
    auto ncols = 3;
    Ar2D<float> X(nrows, ncols);

    { // new scope so I can pretend this is Catch instead of Gtest...
        Ar2D<float> filt(1, 1);
        Ar2D<float> out(nrows, ncols);
        Ar2D<float> ans(nrows, ncols);
        for (int i = 0; i < X.size(); i++) { X.data()[i] = i; }
        filt(0, 0) = 2;
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                ans(i, j) = X(i, j) * 2;
            }
        }

        conv2dx2d_valid_rowmajor(X.data(), nrows, ncols,
            filt.data(), 1, 1, out.data());

        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                EXPECT_EQ(out(i, j), ans(i, j));
            }
        }
    }
    {
        Ar2D<float> filt(2, 2);
        auto nrow_positions = nrows - filt.dimensions()[0] + 1;
        auto ncol_positions = ncols - filt.dimensions()[1] + 1;
        Ar2D<float> out(nrow_positions, ncol_positions);
        Ar2D<float> ans(nrow_positions, ncol_positions);
        for (int i = 0; i < X.size(); i++) { X.data()[i] = i; }
        filt(0, 0) = 2;
        filt(0, 1) = 1;
        filt(1, 0) = 0;
        filt(1, 1) = 3;
        ans(0, 0) = X(0, 0)*filt(0, 0) + X(0, 1)*filt(0, 1) +
            X(1, 0)*filt(1, 0) + X(1, 1)*filt(1, 1);
        ans(0, 1) = X(0, 1)*filt(0, 0) + X(0, 2)*filt(0, 1) +
            X(1, 1)*filt(1, 0) + X(1, 2)*filt(1, 1);

        conv2dx2d_valid_rowmajor(X.data(), nrows, ncols,
            filt.data(), 2, 2, out.data());

        for (int i = 0; i < nrow_positions; i++) {
            for (int j = 0; j < ncol_positions; j++) {
                EXPECT_EQ(out(i, j), ans(i, j)) << "i, j = " << i << ", " << j;
            }
        }
    }
}
TEST(DirectConv, 3dx3d_hwc) {
    auto nrows = 3;
    auto ncols = 3;
    auto nchan = 2;
    Ar3D<float> X(nrows, ncols, nchan);
    auto filt_nrows = 2;
    auto filt_ncols = 2;
    Ar3D<float> filt(filt_nrows, filt_ncols, nchan);
    auto nrow_positions = nrows - filt.dimensions()[0] + 1;
    auto ncol_positions = ncols - filt.dimensions()[1] + 1;
    Ar2D<float> out(nrow_positions, ncol_positions);
    Ar2D<float> ans(nrow_positions, ncol_positions);
    for (int i = 0; i < X.size(); i++) { X.data()[i] = i; }
    for (int i = 0; i < filt.size(); i++) {
        static float vals[8] = {1,0,-1,0, 0,1,0,-1};
        filt.data()[i] = vals[i % 8];
        // filt.data()[i] = (i % 2 ? 1 : -1) * (i % 4 > 1 ? 1 : -1); // 1,-1,-1,1
    }
    // ans(0, 0) = X(0,0,0) - X(0,1,0) + X(1,0,1) - X(1,1,1);
    // ans(0, 1) = X(0,1,0) - X(0,2,0) + X(1,1,1) - X(1,2,1);
    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            ans(i, j) = X(i,j,0) - X(i,1+j,0) + X(1+i,0+j,1) - X(1+i,1+j,1);
        }
    }

    conv3dx3d_valid_hwc(X.data(), nrows, ncols,
        filt.data(), filt_nrows, filt_ncols, nchan, out.data());

    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            EXPECT_EQ(out(i, j), ans(i, j)) << "i, j = " << i << ", " << j;
        }
    }
}
TEST(DirectConv, 3dx3d_chw) {
    auto nrows = 3;
    auto ncols = 3;
    auto nchan = 2;
    Ar3D<float> X(nchan, nrows, ncols);
    auto filt_nrows = 2;
    auto filt_ncols = 2;
    Ar3D<float> filt(filt_nrows, filt_ncols, nchan);
    auto nrow_positions = nrows - filt_nrows + 1;
    auto ncol_positions = ncols - filt_ncols + 1;
    Ar2D<float> out(nrow_positions, ncol_positions);
    Ar2D<float> ans(nrow_positions, ncol_positions);
    for (int i = 0; i < X.size(); i++) { X.data()[i] = i; }
    for (int i = 0; i < filt.size(); i++) {
        static float vals[8] = {1,0,0,-1, 0,1,-1,0};
        filt.data()[i] = vals[i % 8];
    }

    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            ans(i, j) = X(0,i,j) - X(0,1+i,1+j) + X(1,0+i,1+j) - X(1,1+i,0+j);
            // ans(i, j) = X(i,j,0) - X(i,1+j,0) + X(1+i,0+j,1) - X(1+i,1+j,1);
            // printf("ans(%d,%d) = %g\n", i, j, ans(i, j));
        }
    }

    // for (int c = 0; c < nchan; c++) {
    //     for (int i = 0; i < nrows; i++) {
    //         for (int j = 0; j < ncols; j++) {
    //             printf("X(%d,%d,%d) = %g\n", c, i, j, X(c, i, j));
    //         }
    //     }
    // }

    conv3dx3d_valid_chw(X.data(), nrows, ncols,
        filt.data(), filt_nrows, filt_ncols, nchan, out.data());

    for (int i = 0; i < nrow_positions; i++) {
        for (int j = 0; j < ncol_positions; j++) {
            EXPECT_EQ(out(i, j), ans(i, j)) << "i, j = " << i << ", " << j;
        }
    }
}

TEST(CatConv, 2d_nchw_gchw_valid) {
    auto nimgs = 6;
    auto nchan = 4;
    auto nrows = 5;
    auto ncols = 3;
    Ar4D<float> X(nimgs, nchan, nrows, ncols);
    auto nout = 3;
    auto filt_nrows = 2;
    auto filt_ncols = 2;
    Ar4D<float> filt(nout, nchan, filt_nrows, filt_ncols);
    auto nrow_positions = nrows - filt_nrows + 1;
    auto ncol_positions = ncols - filt_ncols + 1;
    Ar4D<float> out(nimgs, nout, nrow_positions, ncol_positions);
    Ar4D<float> ans(nimgs, nout, nrow_positions, ncol_positions);
    for (int i = 0; i < X.size(); i++) { X.data()[i] = i % 7; }
    for (int i = 0; i < filt.size(); i++) {
        static float vals[8] = {1,0,-1,2, -2,3,4,-3};
        filt.data()[i] = vals[i % 8];
    }

    conv2d_nchw_x_gchw_valid(X.data(), nimgs, nchan, nrows, ncols,
        filt.data(), nout, filt_nrows, filt_ncols, out.data());
}

