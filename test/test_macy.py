import numpy as np
import unittest

from macy import Agent, weight, raw_opinion_update_vec


class TestCalculations(unittest.TestCase):

    def setUp(self):

        self.a1_2 = Agent()
        self.a2_2 = Agent()
        self.a3_2 = Agent()
        self.a4_2 = Agent()

        self.a1_2.opinions = np.array([-1.0, 0.5])
        self.a2_2.opinions = np.array([-.5, .2])
        self.a3_2.opinions = np.array([-.7, .6])
        self.a4_2.opinions = np.array([-.7, .4])

        self.a1_3 = Agent()
        self.a2_3 = Agent()
        self.a3_3 = Agent()
        self.a4_3 = Agent()

        self.a1_3.opinions = np.array([-1.0, 0.5, -.7])
        self.a2_3.opinions = np.array([-.5, .2, .8])

        # self.test_network = _make_test_network(2)

    def test_weight(self):

        num = 0.5 + 0.3
        expected = 1 - (num/2.0)
        assert weight(self.a1_2, self.a2_2) == expected

        num = 0.5 + 0.3 + 1.5
        expected = 1 - (num/3.0)
        assert weight(self.a1_3, self.a2_3) == expected

    def test_raw_state_update(self):

        assert raw_opinion_update_vec(
                self.a1_2, [self.a2_2, self.a3_2, self.a4_2]
            ) == np.array([])

    def test_scaled_state_update(self):

        assert False

    def test_polarization(self):

        assert False
