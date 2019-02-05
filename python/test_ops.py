#!/usr/bin/env python

# import tensorflow as tf

# zero_out_module = tf.load_op_library('../lib/zero_out.so')

# with tf.Session(''):
#         print(zero_out_module.zero_out([[1, 2], [3, 4]]).eval())

import tensorflow as tf


class ZeroOutTest(tf.test.TestCase):
    def testZeroOut(self):
        zero_out_module = tf.load_op_library('lib/zero_out.so')
        # v = tf.placeholder(5)
        with self.test_session():
            v = tf.constant([5, 4, 3, 2, 1])
            # v = tf.variable([5, 4, 3, 2, 1])
            # v = v * 2
            # result = zero_out_module.zero_out([5, 4, 3, 2, 1])
            # self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])
            result = zero_out_module.zero_out(v)
            self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])
            # self.assertAllEqual(result.eval(), [10, 0, 0, 0, 0])
            # self.assertAllEqual((v+1).eval(), [10, 7, 7, 7, 7])


if __name__ == "__main__":
    tf.test.main()
