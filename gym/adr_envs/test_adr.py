import unittest
from adr import ADRParam, ADRUniform, ADR 

class TestADRParam(unittest.TestCase):

    def setUp(self):
        self.param = ADRParam(1.0, [0, 2.0], delta=0.02, pq_size=10, boundary_sample_weight=1)
        
    def test_setters_getters(self):
        self.assertEqual(1.0, self.param.get_value())
        self.assertEqual(False, self.param.get_boundary_sample_flag())
        self.param.set_boundary_sample_flag(True)
        self.assertEqual(True, self.param.get_boundary_sample_flag())

        self.assertEqual(1, self.param.get_boundary_sample_weight())
    
    def test_update(self):
        # Initial Value
        self.assertEqual(1.0, self.param.get_value())
        self.assertEqual(0.02, self.param.delta)
        p_thresh = [-1, 1]

        # Update with one full PQ clearly above p_thresh
        for _ in range(10):
            self.param.update(2.0, p_thresh)
        
        self.assertAlmostEqual(1.02, self.param.get_value())
        self.assertEqual(0.02, self.param.delta)

        # Update 10 more full PQs
        for _ in range(100):
            self.param.update(2.0, p_thresh)
        
        self.assertAlmostEqual(1.22, self.param.get_value())
        self.assertEqual(0.02, self.param.delta)

        # Update downwards past min value
        for _ in range(1000):
            self.param.update(-2.0, p_thresh)
        
        self.assertAlmostEqual(0, self.param.get_value())
        self.assertEqual(0.02, self.param.delta)

        # Update arbitrarily large amount of times within PQ thresh
        for _ in range(10000):
            self.param.update(0.5, p_thresh)
        
        self.assertAlmostEqual(0, self.param.get_value())
        self.assertEqual(0.02, self.param.delta)

        # Update upwards past max value
        for _ in range(10000):
            self.param.update(2, p_thresh)
        
        self.assertAlmostEqual(2.0, self.param.get_value())
        self.assertEqual(0.02, self.param.delta)

if __name__ == "__main__":
    unittest.main()