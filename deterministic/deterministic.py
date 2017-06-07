import networkx as nx


def connected_caveman(n_caves=4):
    '''
    Create a caveman graph where every caveman is connected to exactly one
    other caveman from one other cave.
    '''
    n_per_cave = n_caves - 1

    g = nx.caveman_graph(n_caves, n_per_cave)

    lcc = [tuple(el) for el in nx.connected_components(g)]

    for cave_idx, cave in enumerate(lcc[:-1]):
        for el_idx, el in enumerate(cave[cave_idx:]):
            new_edge = (el, lcc[cave_idx+el_idx+1][cave_idx])
            g.add_edge(*new_edge)

    return g
