//
//  test_catconv.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <gtest/gtest.h>
// #include "../src/eigen/Eigen/Core"
#include "../src/eigen/unsupported/Eigen/CXX11/Tensor"

#include "../src/catconv.hpp"

template<class DataT> using Ar2D = Eigen::Tensor<DataT, 2, Eigen::RowMajor>;
template<class DataT> using Ar3D = Eigen::Tensor<DataT, 3, Eigen::RowMajor>;

TEST(CatConv, SmokeTest) {
    auto x = 5;
    EXPECT_EQ(x, 5);
    EXPECT_GT(x, 4);
}

TEST(CatConv, 2d_hw_chw_valid) {
    auto ncard = 4;
    auto nrows = 2;
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
        }
    }

    catconv2d_hw_x_chw_valid(X.data(), nrows, ncols,
        filt.data(), filt_nrows, filt_ncols, ncard, out.data());

    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            EXPECT_EQ(out(i, j), ans(i, j)) << "i, j = " << i << ", " << j;
        }
    }
}
