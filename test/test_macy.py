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
            ((1 + self.a1_2.opinions[0]) * raw_update_vec[0])

        expected_1 = \
            self.a1_2.opinions[1] + \
            ((1 - self.a1_2.opinions[1]) * raw_update_vec[1])

        calculated = opinion_update_vec(
            self.a1_2, neighbors
        )
        expected = np.array([expected_0, expected_1])

        assert (calculated == expected).all(), \
            'calculated: {}\nexpected: {}'.format(calculated, expected)

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

        np.testing.assert_approx_equal(
            expected, calculated,
            err_msg='calc: {}\nexpec: {}'.format(calculated, expected)
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

    # def test_iteration(self):
    #     'Weights and opinions should update as calculated by hand'

    #     a1 = Agent()
    #     a2 = Agent()
    #     a3 = Agent()
    #     a4 = Agent()

    #     a1.opinions = np.array([-1.0, 0.5])
    #     a2.opinions = np.array([-.5, .2])
    #     a3.opinions = np.array([-.7, .6])
    #     a4.opinions = np.array([-.7, .4])

    #     graph = nx.Graph()
    #     graph.add_edges_from([
    #         (e1, e2, {'weight': weight(e1, e2)})
    #         for e1, e2 in [(a1, a2), (a1, a4), (a2, a3), (a3, a4)]
    #     ])

    #     network = Network(initial_graph=graph)

    #     # print([a.opinions for a in network.graph.nodes()])

    #     network.iterate()

    #     # calculation via eq 2a
    #     def getops(i, k):
    #         return ('a{}_2'.format(i)).opinions[k]

    #     # o11 = getops(1, 0); o12 = getops(1, 1)
    #     # o21 = getops(2, 0); o22 = getops(2, 1)
    #     # o31 = getops(3, 0); o32 = getops(3, 1)
    #     # o41 = getops(4, 0); o42 = getops(4, 1)
    #     ea1 = Agent()
    #     ea2 = Agent()
    #     ea3 = Agent()
    #     ea4 = Agent()

    #     ea1.opinions = np.array([-1.0, 0.5])
    #     ea2.opinions = np.array([-.5, .2])
    #     ea3.opinions = np.array([-.7, .6])
    #     ea4.opinions = np.array([-.7, .4])

    #     o11 = ea1.opinions[0]; o12 = ea1.opinions[1]
    #     o21 = ea2.opinions[0]; o22 = ea2.opinions[1]
    #     o31 = ea3.opinions[0]; o32 = ea3.opinions[1]
    #     o41 = ea4.opinions[0]; o42 = ea4.opinions[1]

    #     # equivalent to delta-s from eq 2
    #     f = .25
    #     d11 = f*.06   ; d12 = f*(-.26)
    #     d21 = f*.16   ; d22 = f*.1
    #     d31 = f*(-.14); d32 = f*.1
    #     d41 = f*(-.24); d42 = f*(-.26)

    #     e11 = o11 + (d11*(1 + o11)); e12 = o12 + (d12*(1 - o12))
    #     e21 = o21 + (d21*(1 + o21)); e22 = o22 + (d22*(1 - o22))
    #     e31 = o31 + (d31*(1 + o31)); e32 = o32 + (d32*(1 - o32))
    #     e41 = o41 + (d41*(1 + o41)); e42 = o42 + (d42*(1 - o42))

    #     expected_opinions = set([
    #         (e11, e12),
    #         (e21, e22),
    #         (e31, e32),
    #         (e41, e42)
    #     ])

    #     calculated_opinions = set(
    #         tuple(agent.opinions) for agent in network.graph.nodes()
    #     )

    #     assert expected_opinions == calculated_opinions, \
    #         'expected: {}\ncalculated: {}'.format(
    #             expected_opinions, calculated_opinions
    #         )
