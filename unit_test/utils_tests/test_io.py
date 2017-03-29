#!/usr/bin/env python
# encoding: utf-8

import unittest
from utils import io
class UtilsIOTest(unittest.TestCase):
    def test_slice2batches(self):
        input_list = [1,2,3]*10
        batch_size = 3
        ret = io.slice2batches(input_list, batch_size)
        self.assertEqual(10, len(ret))
        for li in ret:
            self.assertEqual([1,2,3], li)
