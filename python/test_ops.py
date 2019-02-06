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

    def testCatConv(self):
        with self.test_session():
            v = tf.constant([[5., 4], [3, 2]])
            result = self.op_module.dummy_float(v, v + 2)
            answer = np.arange(v.eval().size).reshape(v.shape)
            self.assertAllEqual(result.eval(), answer)


# class CatConvTest(tf.test.TestCase):


if __name__ == "__main__":
    tf.test.main()
