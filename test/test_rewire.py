'''
Checking that rewiring works as expected.

See DellaPosta, D., Shi, Y., & Macy, M. (2015).  Why Do Liberals Drink Lattes?
American Journal of Sociology, p. 1490-1491
'''
import networkx as nx
import numpy as np
import unittest

from copy import deepcopy
from nose.tools import eq_, ok_

from macy import (
    Agent, weight, raw_opinion_update_vec, opinion_update_vec, polarization,
    Network
)


class TestRewiring(unittest.TestCase):

    def setUp(self):
        pass

    def test_connected_caveman_rewire(self):
        '''
        Rewiring should conserve number of neighbors for all agents and swap some edges.
        '''
        # Initialize graph to be copied and randomized.
        cc = nx.connected_caveman_graph(10, 5)
        num_base = [sum(1 for _ in nx.all_neighbors(cc, n))
                    for n in cc.nodes()]

        # Create networks and check that rewire conserves each agent's
        # number of neighbors. Since it's random it took a few times running
        # the tests to see problems. This at least says it's reliable
        # twenty times in a row, whatever that's worth.
        for _ in range(20):
            for cxn_prob in np.arange(0.1, 0.5, 0.1):

                net = Network(deepcopy(cc))
                net.add_random_connections(cxn_prob)

                num_rand = [sum(1 for _ in nx.all_neighbors(net.graph, n))
                            for n in net.graph.nodes()]

                # Check that all neighbor counts are conserved.
                err_msg = '\n{} != \n{} for cxn_prob={}'.format(
                    num_base, num_rand, cxn_prob
                )
                eq_(num_base, num_rand, err_msg)

                # Check that at least some edges were changed. TODO check exact N
                num_changed = len(
                    set(cc.edges()).difference(net.graph.edges())
                )
                err_msg = 'Set difference between {} and {} empty' \
                          'for cxn_prob={}'.format(
                              cc.edges(), net.graph.edges(), cxn_prob
                           )
                ok_(num_changed > 0, err_msg)
