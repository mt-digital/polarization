import networkx as nx
import numpy as np
import unittest

from macy import (
    Agent, weight, raw_opinion_update_vec, opinion_update_vec, polarization,
    Network
)


class TestBasicCalculations(unittest.TestCase):

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

        num_neighbors_fac = 1.0 / (2.0 * 3)
        w_12 = weight(self.a1_2, self.a2_2)
        w_13 = weight(self.a1_2, self.a3_2)
        w_14 = weight(self.a1_2, self.a4_2)

        S = (w_12*np.array([.5, -.3])) + \
            (w_13*np.array([.3, .1])) + \
            (w_14*np.array([.3, -.1]))

        expected = num_neighbors_fac * S
        calculated = raw_opinion_update_vec(
            self.a1_2, [self.a2_2, self.a3_2, self.a4_2]
        )

        assert (calculated == expected).all(), 'calc: {}\nexpec: {}'.format(
            calculated, expected
        )

    def test_scaled_state_update(self):
        neighbors = [self.a2_2, self.a3_2, self.a4_2]

        raw_update_vec = raw_opinion_update_vec(self.a1_2, neighbors)

        expected_0 = \
            self.a1_2.opinions[0] + \
            ((1 - self.a1_2.opinions[0]) * raw_update_vec[0])

        expected_1 = \
            self.a1_2.opinions[1] + \
            ((1 - self.a1_2.opinions[1]) * raw_update_vec[1])

        calculated = opinion_update_vec(
            self.a1_2, neighbors
        )

        assert (calculated == np.array([expected_0, expected_1])).all()

    def test_polarization(self):

        d = np.zeros((4, 4))

        d[0, 1] = .4     ; d[0, 2] = .2     ; d[0, 3] = .2
        d[1, 0] = d[0, 1]; d[1, 2] = .3     ; d[1, 3] = .2
        d[2, 0] = d[0, 2]; d[2, 1] = d[1, 2]; d[2, 3] = .1
        d[3, 0] = d[0, 3]; d[3, 1] = d[1, 3]; d[3, 2] = d[2, 3]

        d_mean = d.sum() / (4 * 3)

        d_sub_mean = d - d_mean
        for i in range(4):
            d_sub_mean[i, i] = 0.0

        print(d_sub_mean)
        expected = np.sum((d_sub_mean)**2) / (4 * 3)

        g = nx.Graph()
        a1 = self.a1_2
        a2 = self.a2_2
        a3 = self.a3_2
        a4 = self.a4_2

        g.add_edges_from([
            (e1, e2, {'weight': weight(e1, e2)})
            for e1, e2 in [(a1, a2), (a1, a4), (a2, a3), (a3, a4)]
        ])

        calculated = polarization(g)

        assert expected == calculated, 'calc: {}\nexpec: {}'.format(
            calculated, expected
        )


class TestNetworkIterations(unittest.TestCase):

    def setUp(self):

        self.a1_2 = Agent()
        self.a2_2 = Agent()
        self.a3_2 = Agent()
        self.a4_2 = Agent()

        a1 = self.a1_2
        a2 = self.a2_2
        a3 = self.a3_2
        a4 = self.a4_2

        self.a1_2.opinions = np.array([-1.0, 0.5])
        self.a2_2.opinions = np.array([-.5, .2])
        self.a3_2.opinions = np.array([-.7, .6])
        self.a4_2.opinions = np.array([-.7, .4])

        self.graph = nx.Graph()
        self.graph.add_edges_from([
            (e1, e2, {'weight': weight(e1, e2)})
            for e1, e2 in [(a1, a2), (a1, a4), (a2, a3), (a3, a4)]
        ])

    def test_add_random_connections(self):
        'Random connections should increase number of edges to saturation'
        network = Network(initial_graph=self.graph)
        for _ in range(2):
            cur_edges = network.graph.edges()
            network.add_random_connection()
            assert len(cur_edges) + 1 == len(network.graph.edges())

        with self.assertRaises(RuntimeError):
            network.add_random_connection()

    def test_iteration(self):
        'Weights and opinions should update as calculated by hand'

        network = Network(initial_graph=self.graph)

        network.iterate()

        assert network is None
