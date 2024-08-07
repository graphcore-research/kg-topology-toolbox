# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Utility functions
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import coo_array, csc_array, csr_array


def jaccard_similarity(
    entities_1: NDArray[np.int32], entities_2: NDArray[np.int32]
) -> float:
    """
    Jaccard Similarity function for two sets of entities.

    :param entities_1: the array of IDs for the first set of entities.
    :param entities_2: the array of IDs for the second set of entities.

    :return: Jaccard Similarity score for two sets of entities.
    """
    intersection = len(np.intersect1d(entities_1, entities_2))
    union = len(entities_1) + len(entities_2) - intersection
    return float(intersection / union)


def _composition_count_worker(
    adj_csr: csr_array, adj_csc: csc_array, tail_shift: int = 0
) -> pd.DataFrame:
    adj_2hop = adj_csr @ adj_csc
    adj_composition = (adj_2hop.tocsc() * (adj_csc > 0)).tocoo()
    df_composition = pd.DataFrame(
        dict(
            h=adj_composition.row,
            t=adj_composition.col + tail_shift,
            n_triangles=adj_composition.data,
        )
    )
    return df_composition


def composition_count(
    df: pd.DataFrame, chunk_size: int, workers: int, directed: bool = True
) -> pd.DataFrame:
    """A helper function to compute the composition count of a graph.

    :param df: A graph represented as a pd.DataFrame. Must contain the columns
        `h` and `t`. No self-loops should be present in the graph.
    :param chunk_size: Size of chunks of columns of the adjacency matrix to be
        processed together.
    :param workers: Number of workers processing chunks concurrently
    :param directed: Boolean flag. If false, bidirectional edges are considered for
        triangles by adding the adjacency matrix and its transposed. Defaults to True.

    :return: The results dataframe. Contains the following columns:

        - **h** (int): Index of the head entity.
        - **t** (int): Index of the tail entity.
        - **n_triangles** (int): Number of compositions for the (h, t) edge.
    """

    n_nodes = max(df[["h", "t"]].max()) + 1
    adj = coo_array(
        (np.ones(len(df)), (df.h, df.t)),
        shape=[n_nodes, n_nodes],
    ).astype(np.uint16)
    if not directed:
        adj = adj + adj.T
    n_cols = adj.shape[1]
    adj_csr = adj.tocsr()
    adj_csc = adj.tocsc()
    adj_csc_slices = {
        i: adj_csc[:, i * chunk_size : min((i + 1) * chunk_size, n_cols)]
        for i in range(int(np.ceil(n_cols / chunk_size)))
    }

    if len(adj_csc_slices) > 1 and workers > 1:
        with Pool(workers) as pool:
            df_composition_list = pool.starmap(
                _composition_count_worker,
                (
                    (adj_csr, adj_csc_slice, i * chunk_size)
                    for i, adj_csc_slice in adj_csc_slices.items()
                ),
            )
    else:
        df_composition_list = [
            _composition_count_worker(adj_csr, adj_csc_slice, i * chunk_size)
            for i, adj_csc_slice in adj_csc_slices.items()
        ]

    return pd.concat(df_composition_list)
