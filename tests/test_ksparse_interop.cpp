//
//  test_ksparse_interop.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "../src/ksparse_interop.hpp"
#include "../src/ksparse_interop_grad.hpp"

#include <gtest/gtest.h>
#include "testing_utils.hpp"

template<class DataT>
void populate_x(Ar5D<DataT>& X, DataT maxval) {
    // argmax in each group should be at idx (n + i + j) % group_sz
    // we also set all the other indices to a constant so that we can
    // check that the output is correct easily
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

TEST(KSparseInterop, Forward) {
    const int log2_group_sz = 2;
    const int group_sz = 1 << log2_group_sz;
    auto ngroups = 1;
    // using dtype = int16_t;
    using dtype = float;

    auto nimgs = 6;
    auto nrows = 5;
    auto ncols = 4;
    // auto nimgs = 1;
    // auto nrows = 1;
    // auto ncols = 1;
    auto nchan = ngroups * group_sz;
    Ar5D<dtype> X(nimgs, nrows, ncols, ngroups, group_sz);
    Ar4D<dtype> X_packed(nimgs, nrows, ncols, ngroups);
    Ar5D<dtype> X_hat(nimgs, nrows, ncols, ngroups, group_sz);

    dtype maxval = 9 << log2_group_sz; // so low bits don't get overwritten
    populate_x(X, maxval);

    dense2sparse_nhwc<log2_group_sz>(X.data(), nimgs, nrows, ncols, nchan, X_packed.data());
    sparse2dense_nhwc(X_packed.data(), nimgs, nrows, ncols, ngroups, log2_group_sz, X_hat.data());
    // sparse2dense_nhwc<log2_group_sz>(X_packed.data(), nimgs, nrows, ncols, ngroups, X_hat.data());

    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                for (int c = 0; c < ngroups; c++) {
                    auto argmax = (n + i + j) % group_sz;
                    for (int cc = 0; cc < group_sz; cc++) {
                        // target is 0 for non-max entries, even though X has
                        // nonzero, cuz it's supposed to be lossy
                        auto target = (cc == argmax) ? maxval : 0;
                        EXPECT_EQ(X_hat(n, i, j, c, cc), target)
                            << "cc=" << cc << " packed_val=" << X_packed(n, i, j, c);
                    }
                }
            }
        }
    }
}

TEST(KSparseInterop, Backward) {
    const int log2_group_sz = 2;
    const int group_sz = 1 << log2_group_sz;
    auto ngroups = 1;
    using dtype = float;

    auto nimgs = 6;
    auto nrows = 5;
    auto ncols = 4;
    auto nchan = ngroups * group_sz;
    Ar5D<dtype> X(nimgs, nrows, ncols, ngroups, group_sz);
    Ar4D<dtype> errs(nimgs, nrows, ncols, ngroups);
    Ar5D<dtype> X_hat(nimgs, nrows, ncols, ngroups, group_sz);
    Ar4D<dtype> errs2(nimgs, nrows, ncols, ngroups);

    dtype maxval = 9 << log2_group_sz; // so low bits don't get overwritten
    populate_x(X, maxval);

    // use ksparse packed X as errs
    dense2sparse_nhwc<log2_group_sz>(
        X.data(), nimgs, nrows, ncols, nchan, errs.data());
    dense2sparse_nhwc_grad(errs.data(), nimgs, nrows, ncols, ngroups,
        log2_group_sz, X_hat.data());

    // backward grad should just be a sparse2dense operation
    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                for (int c = 0; c < ngroups; c++) {
                    auto argmax = (n + i + j) % group_sz;
                    for (int cc = 0; cc < group_sz; cc++) {
                        // target is 0 for non-max entries, even though X has
                        // nonzero, cuz it's supposed to be lossy
                        auto target = (cc == argmax) ? maxval : 0;
                        EXPECT_EQ(X_hat(n, i, j, c, cc), target)
                            << "cc=" << cc << " packed_val=" << errs(n, i, j, c);
                    }
                }
            }
        }
    }

    // now make X_hat be sparse2dense of errs, and compute grads for that
    sparse2dense_nhwc(errs.data(), nimgs, nrows, ncols, ngroups,
        log2_group_sz, X_hat.data());
    sparse2dense_nhwc_grad(errs.data(), nimgs, nrows, ncols, ngroups,
        log2_group_sz, X_hat.data(), errs2.data());

    // sparse2dense grad should invert dense2sparse, given access to sparse in
    for (int n = 0; n < nimgs; n++) {
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                for (int c = 0; c < ngroups; c++) {
                    EXPECT_EQ(errs(n, i, j, c), errs2(n, i, j, c));                }
            }
        }
    }
}
