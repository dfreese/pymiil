import unittest
import numpy as np
import miil
from numpy.testing import assert_array_equal

class TestListMode(unittest.TestCase):
    def setUp(self):
        self.events = np.zeros(10, dtype=miil.eventcoinc_dtype)
        self.events['crystal1'] = np.arange(0, 10)
        self.lors = miil.coinc_to_lor(self.events)
        self.vec = miil.create_sparse_column_vector(self.lors)

    def test_from_vec(self):
        data = miil.create_listmode_from_vec(self.vec)
        self.assertEqual(data.dtype, miil.cudarecon_type1_vec_dtype)
        assert_array_equal(data['weight'], np.ones(data.size))
        assert_array_equal(data['weight1'], np.zeros(data.size))

    def test_from_lors(self):
        data = miil.create_listmode_from_lors(self.lors)
        self.assertEqual(data.dtype, miil.cudarecon_type0_vec_dtype)
        assert_array_equal(data['randoms_est'], np.zeros(data.size))
        assert_array_equal(data['tof_scatter_est'], np.zeros(data.size))
        assert_array_equal(data['scatter_est'], np.zeros(data.size))

    def test_from_lors_type1(self):
        data = miil.create_listmode_from_lors(self.lors, list_type=1)
        self.assertEqual(data.dtype, miil.cudarecon_type1_vec_dtype)
        assert_array_equal(data['weight'], np.ones(data.size))
        assert_array_equal(data['weight1'], np.zeros(data.size))

    def test_from_coinc(self):
        data = miil.create_listmode_data(self.events)
        self.assertEqual(data.dtype, miil.cudarecon_type0_vec_dtype)
        assert_array_equal(data['randoms_est'], np.zeros(data.size))
        assert_array_equal(data['tof_scatter_est'], np.zeros(data.size))
        assert_array_equal(data['scatter_est'], np.zeros(data.size))

    def test_from_coinc_type1(self):
        data = miil.create_listmode_data(self.events, list_type=1)
        self.assertEqual(data.dtype, miil.cudarecon_type1_vec_dtype)
        assert_array_equal(data['weight'], np.ones(data.size))
        assert_array_equal(data['weight1'], np.zeros(data.size))
