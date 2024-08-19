# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Utility functions
"""

import warnings
from collections.abc import Iterable
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_integer_dtype
from scipy.sparse import coo_array, csc_array, csr_array


def check_kg_df_structure(kg_df: pd.DataFrame, h: str, r: str, t: str) -> None:
    """
    Utility to perform sanity checks on the structure of the provided DataFrame,
    to ensure that it encodes a Knowledge Graph in a compatible way.

    :param kg_df:
        The Knowledge Graph DataFrame.
    :param h:
        The name of the column with the IDs of head entities.
    :param r:
        The name of the column with the IDs of relation types.
    :param t:
        The name of the column with the IDs of tail entities.

    """
    # check h,r,t columns are present and of an integer type
    for col_name in [h, r, t]:
        if col_name in kg_df.columns:
            if not is_integer_dtype(kg_df[col_name]):
                raise TypeError(f"Column {col_name} needs to be of an integer dtype")
        else:
            raise ValueError(f"DataFrame {kg_df} has no column named {col_name}")
    # check there are no duplicated (h,r,t) triples
    if kg_df[[h, r, t]].duplicated().any():
        warnings.warn(
            "The Knowledge Graph contains duplicated edges"
            " -- some functionalities may produce incorrect results"
        )


def node_degrees_and_rels(
    df: pd.DataFrame, column: str, n_entity: int, return_relation_list: bool
) -> pd.DataFrame:
    """
    Aggregate edges by head/tail node and compute associated statistics.

    :param df:
        Dataframe of (h,r,t) triples.
    :param column:
        Name of the column used to aggregate edges.
    :param n_entity:
        Total number of entities in the graph.
    :param return_relation_list:
        If True, return the list of unique relations types
        in the set of aggregated edges.

    :return:
        The result DataFrame, indexed on the IDs of the graph entities,
        with columns:

        - **degree** (int): Number of triples in the aggregation.
        - **unique_rel** (int): Number of distinct relation types
            in the set of aggregated edges.
        - **rel_list** (Optional[list]): List of unique relation types
            in the set of aggregated edges.
            Only returned if `return_relation_list = True`.
    """
    rel_list = {"rel_list": ("r", "unique")} if return_relation_list else {}
    deg_df = pd.DataFrame(
        df.groupby(column).agg(
            degree=("r", "count"), unique_rel=("r", "nunique"), **rel_list  # type: ignore
        ),
        index=np.arange(n_entity),
    )
    deg_df[["degree", "unique_rel"]] = (
        deg_df[["degree", "unique_rel"]].fillna(0).astype(int)
    )
    return deg_df


def aggregate_by_relation(edge_topology_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate topology metrics of all triples of the same relation type.
    To be applied to a DataFrame of metrics having at least columns
    `h`, `r`, `t` (e.g., the output of
    :meth:`KGTopologyToolbox.edge_degree_cardinality_summary` or
    :meth:`KGTopologyToolbox.edge_pattern_summary`).

    The returned dataframe is indexed over relation type IDs, with columns
    giving the aggregated statistics of triples of the corresponding relation.
    The name of the columns is of the form ``column_name_in_input_df + suffix``.
    The aggregation is performed by returning:

    - for numerical metrics: mean, standard deviation and quartiles
      (``suffix`` = "_mean", "_std", "_quartile1", "_quartile2", "_quartile3");
    - for boolean metrics: the fraction of triples of the relation type
      with metric = True (``suffix`` = "_frac");
    - for string metrics: for each possible label, the fraction of triples
      of the relation type with that metric value (``suffix`` = "_{label}_frac")
    - for list metrics: the unique metric values across triples of the relation
      type (``suffix`` = "_unique").

    :param edge_topology_df:
        pd.DataFrame of edge topology metrics.
        Must contain at least three columns `h`, `r`, `t`.

    :return:
        The results dataframe. In addition to the columns with the aggregated
        metrics by relation type, it also contains columns:

        - **num_triples** (int): Number of triples for each relation type.
        - **frac_triples** (float): Fraction of overall triples represented by each
          relation type.
        - **unique_h** (int): Number of unique head entities used by triples of each
          relation type.
        - **unique_t** (int): Number of unique tail entities used by triples of each
          relation type.
    """
    df_by_r = edge_topology_df.groupby("r")
    df_res = df_by_r.agg(num_triples=("r", "count"))
    df_res["frac_triples"] = df_res["num_triples"] / edge_topology_df.shape[0]
    col: str
    for col, col_dtype in edge_topology_df.drop(columns=["r"]).dtypes.items():  # type: ignore
        if col in ["h", "t"]:
            df_res[f"unique_{col}"] = df_by_r[col].nunique()
        elif col_dtype == object:
            if isinstance(edge_topology_df[col].iloc[0], str):
                for label in np.unique(edge_topology_df[col]):
                    df_res[f"{col}_{label}_frac"] = (
                        edge_topology_df[edge_topology_df[col] == label]
                        .groupby("r")[col]
                        .count()
                        / df_res["num_triples"]
                    ).fillna(0)
            elif isinstance(edge_topology_df[col].iloc[0], Iterable):
                df_res[f"{col}_unique"] = (
                    df_by_r[col]
                    .agg(np.unique)
                    .apply(
                        lambda x: (
                            np.unique(
                                np.concatenate(
                                    [lst for lst in x if len(lst) > 0] or [[]]
                                )
                            ).tolist()
                        )
                    )
                )
            else:
                print(f"Skipping column {col}: no known aggregation mode")
                continue
        elif col_dtype == int or col_dtype == float:
            df_res[f"{col}_mean"] = df_by_r[col].mean()
            df_res[f"{col}_std"] = df_by_r[col].std()
            for q in range(1, 4):
                df_res[f"{col}_quartile{q}"] = df_by_r[col].agg(
                    lambda x: np.quantile(x, 0.25 * q)
                )
        elif col_dtype == bool:
            df_res[f"{col}_frac"] = df_by_r[col].mean()
    return df_res


def jaccard_similarity(
    entities_1: NDArray[np.int32], entities_2: NDArray[np.int32]
) -> float:
    """
    Jaccard Similarity function for two sets of entities.

    :param entities_1:
        Array of IDs for the first set of entities.
    :param entities_2:
        Array of IDs for the second set of entities.

    :return:
        Jaccard Similarity score for two sets of entities.
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

    :param df:
        A graph represented as a pd.DataFrame. Must contain the columns
        `h` and `t`. No self-loops should be present in the graph.
    :param chunk_size:
        Size of chunks of columns of the adjacency matrix to be
        processed together.
    :param workers:
        Number of workers processing chunks concurrently
    :param directed:
        Boolean flag. If false, bidirectional edges are considered for
        triangles by adding the adjacency matrix and its transposed. Default: True.

    :return:
        The results dataframe. Contains the following columns:
        - **h** (int): Index of the head entity.
        - **t** (int): Index of the tail entity.
        - **n_triangles** (int): Number of compositions for the (h, t) edge.
    """

    n_nodes = df[["h", "t"]].max().max() + 1
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
