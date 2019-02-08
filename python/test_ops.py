#!/usr/bin/env python

# import tensorflow as tf

# zero_out_module = tf.load_op_library('../lib/zero_out.so')

# with tf.Session(''):
#         print(zero_out_module.zero_out([[1, 2], [3, 4]]).eval())

import numpy as np
import tensorflow as tf


class ExampleOpsTest(tf.test.TestCase):

    def setUp(self):
        self.op_module = tf.load_op_library('lib/example_ops.so')

    def testZeroOut(self):
        with self.test_session():
            v = tf.constant([5, 4, 3, 2, 1])
            result = self.op_module.zero_out(v)
            self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

    def testDummyFloat(self):
        with self.test_session():
            v = tf.constant([[5., 4], [3, 2]])
            result = self.op_module.dummy_float(v, v + 2)
            answer = np.arange(v.eval().size).reshape(v.shape)
            self.assertAllEqual(result.eval(), answer)


class CatConvTest(tf.test.TestCase):

    def setUp(self):
        self.op_module = tf.load_op_library('lib/catconv_ops.so')

    def testCatConv(self):
        with self.test_session():
            n, c, h, w = 2, 6, 5, 4
            g, k, ll = 2, 3, 3
            X = np.random.randint(0, 16, size=(n, c, h, w), dtype=np.int32)
            filt = np.random.randn(g, c, k, ll).astype(np.float32)
            X, filt = tf.constant(X), tf.constant(filt)
            result = self.op_module.cat_conv(X, filt).eval()
            out_shape = result.shape
            self.assertAllEqual(out_shape, (n, g, h - 1, w - 1))


if __name__ == "__main__":
    tf.test.main()
