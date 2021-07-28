import numpy as np
if __name__ == '__main__':
    edges=list()
    edges.append([1, 3] )
    edges.append([3, 5])
    edges.append([5, 7])
    edges = np.asarray(edges)
    print(edges)
    edges = np.sort(edges, axis=0)
    nodes=np.sort(np.unique(edges.flatten()))

    print(nodes.shape)

    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])

    link_idx = mapping[edges]

    print(link_idx)