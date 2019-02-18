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

// NOTE: this is not the same as the populate_X in test_ksparse_interop
template<class DataT>
void populate_x(Ar5D<DataT>& X, DataT maxval, bool fixed_argmax=false) {
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
                    auto argmax = (n + i + j + c) % group_sz;
                    argmax = fixed_argmax ? group_sz - 1 : argmax;
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
    const uint16_t out_log2_group_sz = 3;
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
                    // make sure that packed vals are okay
                    auto target = filt_nrows * filt_ncols * in_ngroups * maxval;
                    dtype y;
                    uint16_t idx;
                    unpack_idx_val(Y_packed(n, i, j, c),
                        out_log2_group_sz, &idx, &y);
                    EXPECT_EQ(y, target);
                    // all vals same, so highest idx should yield packed max
                    EXPECT_EQ(idx, out_group_sz - 1);

                    for (int cc = 0; cc < out_group_sz; cc++) {
                        target = filt_nrows * filt_ncols * in_ngroups * maxval;
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


TEST(KSparseConv, Backward) {
    const int in_log2_group_sz = 2;
    // const int in_log2_group_sz = 1;
    const int in_nbits = in_log2_group_sz;
    const int in_group_sz = 1 << in_log2_group_sz;
    uint16_t in_ngroups = 1;
    auto in_nchan = in_ngroups * in_group_sz;
    // using dtype = int16_t;
    using dtype = float;

    auto nimgs = 6;
    auto in_nrows = 8;
    auto in_ncols = 7;
    // auto nimgs = 1;
    // auto in_nrows = 3;
    // auto in_ncols = 3;
    // auto in_nrows = 4;
    // auto in_ncols = 3;

    auto filt_nrows = 3;
    auto filt_ncols = 3;
    auto out_ngroups = 2;
    // auto out_ngroups = 1;
    dtype filt_val = 2;
    const uint16_t out_log2_group_sz = 2;
    // const uint16_t out_log2_group_sz = 1;
    uint16_t out_group_sz = 1 << out_log2_group_sz;
    auto out_nrows = in_nrows - filt_nrows + 1;
    auto out_ncols = in_ncols - filt_ncols + 1;

    Ar5D<dtype> X(nimgs, in_nrows, in_ncols, in_ngroups, in_group_sz);
    Ar4D<dtype> X_packed(nimgs, in_nrows, in_ncols, in_ngroups);
    Ar4D<dtype> Y_packed(nimgs, out_nrows, out_ncols, out_ngroups);
    // Ar5D<dtype> Y_hat(nimgs, out_nrows, out_ncols, out_ngroups, out_group_sz);

    Ar6D<dtype> filt(out_ngroups, out_group_sz, filt_nrows, filt_ncols, in_ngroups, in_group_sz);
    filt.setConstant(filt_val);
    // filt.setConstant(1);

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
    populate_x(X, maxval, /* fixed_argmax */ true);
    dense2sparse_nhwc<in_log2_group_sz>(X.data(), nimgs, in_nrows,
        in_ncols, in_nchan, X_packed.data());

    sparse2sparse_conv2d_nhwc_x_ghwc_valid(
        X_packed.data(), nimgs, in_nrows, in_ncols,
        filt.data(), out_ngroups, out_log2_group_sz, filt_nrows, filt_ncols,
        in_ngroups, in_log2_group_sz, Y_packed.data());

    Ar4D<dtype> errs(Y_packed);
    Ar6D<dtype> filt_grad(filt);
    Ar4D<dtype> in_grad(X_packed);
    // initilize to nonzeros to hopefully screw up results if code is wrong
    filt_grad.setRandom<Eigen::internal::NormalRandomGenerator<dtype> >();
    in_grad.setRandom<Eigen::internal::NormalRandomGenerator<dtype> >();

    sparse2sparse_conv2d_nhwc_x_ghwc_valid_grad(
        X_packed.data(), nimgs, in_nrows, in_ncols,
        filt.data(), out_ngroups, out_log2_group_sz, filt_nrows, filt_ncols,
        in_ngroups, in_log2_group_sz,
        errs.data(), filt_grad.data(), in_grad.data());

    auto errval = filt_nrows * filt_ncols * in_ngroups *
        maxval * filt_val;
    auto target_grad_per_filt_position = errval * out_ngroups * filt_val;
    auto max_target_grad = target_grad_per_filt_position * filt_nrows * filt_ncols;
    // printf("target errval, grad not at edges: %g, %g\n", errval, max_target_grad);

    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < in_nrows; i++) {
            for (int j = 0; j < in_ncols; j++) {
                for (int c = 0; c < in_ngroups; c++) {

                    auto row_dist_to_start_edge = i;
                    auto row_dist_to_end_edge = (in_nrows - 1) - i;
                    auto row_dist_to_edge = MIN(row_dist_to_start_edge, row_dist_to_end_edge);
                    auto col_dist_to_start_edge = j;
                    // auto col_dist_to_end_edge = MAX(0, in_ncols - filt_ncols - i);
                    auto col_dist_to_end_edge = (in_ncols - 1) - j;
                    auto col_dist_to_edge = MIN(col_dist_to_start_edge, col_dist_to_end_edge);
                    auto row_multiplier = MIN(filt_nrows, row_dist_to_edge + 1);
                    auto col_multiplier = MIN(filt_ncols, col_dist_to_edge + 1);
                    row_multiplier = MIN(row_multiplier, out_nrows);
                    col_multiplier = MIN(col_multiplier, out_ncols);
                    auto grad_multiplier = row_multiplier * col_multiplier;
                    auto target_grad = target_grad_per_filt_position * grad_multiplier;

                    // printf("------------------------\n");
                    // printf("row_multiplier, col_multiplier = %d, %d\n", row_multiplier, col_multiplier);

                    dtype x_grad;
                    uint16_t in_idx;
                    unpack_idx_val(in_grad(n, i, j, c), in_nbits, &in_idx, &x_grad);
                    // EXPECT_NEAR(x_grad, target_grad, 1.0) << " n,i,j,c=(" <<
                    EXPECT_EQ(x_grad, target_grad) << " n,i,j,c=(" <<
                        n << "," << i << "," << j << "," << c << ")";

                    // std::cout << " n,i,j,c=(" << n << "," << i << "," << j << "," << c << ")";
                    // printf("(%d,%d,%d,%d): target=%g, actual=%g\n", n, i, j, c, target_grad, x_grad);

                    // this is same as forward test; just making explicit
                    // what errs are so that above check makes sense; in
                    // particular, shows that each element of errs has value
                    // given below
                    if (i >= out_nrows || j >= out_ncols ||
                        c >= out_ngroups) { continue; }
                    dtype y;
                    uint16_t idx;
                    unpack_idx_val(errs(n, i, j, c),
                        out_log2_group_sz, &idx, &y);
                    EXPECT_EQ(y, errval);
                    EXPECT_EQ(idx, out_group_sz - 1);
                }
            }
        }
    }

    auto grad_f_target = errval * maxval * nimgs * out_nrows * out_ncols;
    // printf("target filter grad = %g\n", grad_f_target);
    for (int g = 0; g < out_ngroups; g++) {
        for (int gg = 0; gg < out_group_sz; gg++) {
            for (int ii = 0; ii < filt_nrows; ii++) {
                for (int jj = 0; jj < filt_ncols; jj++) {
                    for (int c = 0; c < in_ngroups; c++) {
                        auto argmax = in_group_sz - 1;
                        for (int cc = 0; cc < in_group_sz; cc++) {
                            auto grad = filt_grad(g, gg, ii, jj, c, cc);

                            // only last neuron in group will fire
                            auto in_should_fire = cc == argmax;
                            auto out_should_fire = gg == (out_group_sz - 1);
                            auto should_have_grad = in_should_fire && out_should_fire;
                            auto target = should_have_grad ? grad_f_target : 0;
                            // printf("(%d,%d,%d,%d,%d,%d): grad=%g\n", g, gg, ii, jj, c, cc, grad);
                            EXPECT_EQ(grad, target);
                        }
                    }
                }
            }
        }
    }
}
