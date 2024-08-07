# -*- coding: utf-8 -*-
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Topology toolbox main functionalities
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from scipy.sparse import coo_array

from kg_topology_toolbox.utils import composition_count, jaccard_similarity


class KGTopologyToolbox:
    """
    Toolbox class to compute Knowledge Graph topology statistics.
    """

    def __init__(
        self,
        kg_df: pd.DataFrame,
        head_column: str = "h",
        relation_column: str = "r",
        tail_column: str = "t",
    ):
        """
        Instantiate the Topology Toolbox for a Knowledge Graph defined
        by the list of its edges (h,r,t).

        :param kg_df:
            A Knowledge Graph represented as a pd.DataFrame.
            Must contain at least three columns, which specify the IDs of
            head entity, relation type and tail entity for each edge.
        :param head_column:
            The name of the column with the IDs of head entities. Default: "h".
        :param head_column:
            The name of the column with the IDs of relation types. Default: "r".
        :param head_column:
            The name of the column with the IDs of tail entities. Default: "t".

        """
        for col_name in [head_column, relation_column, tail_column]:
            if col_name in kg_df.columns:
                if not is_integer_dtype(kg_df[col_name]):
                    raise TypeError(
                        f"Column {col_name} needs to be of an integer dtype"
                    )
            else:
                raise ValueError(f"DataFrame {kg_df} has no column named {col_name}")
        self.df = kg_df[[head_column, relation_column, tail_column]].rename(
            columns={head_column: "h", relation_column: "r", tail_column: "t"}
        )
        self.n_entity = self.df[["h", "t"]].max().max() + 1
        self.n_rel = self.df.r.max() + 1

    def node_degree_summary(self, return_relation_list: bool = False) -> pd.DataFrame:
        """
        For each entity, this function computes the number of edges having it as a head
        (head-degree, or out-degree), as a tail (tail-degree, or in-degree)
        or one of the two (total-degree) in the Knowledge Graph.
        The in-going and out-going relation types are also identified.

        The output dataframe is indexed on the IDs of the graph entities.

        :param return_relation_list:
            If True, return the list of unique relations going
            in/out of an entity. WARNING: expensive for large graphs.

        :return:
            The results dataframe, indexed over the same entity ID `e` used in df,
            with columns:

            - **h_degree** (int): Number of triples with head entity `e`.
            - **t_degree** (int): Number of triples with tail entity `e`.
            - **tot_degree** (int): Number of triples with head entity `e` or tail entity `e`.
            - **h_unique_rel** (int): Number of distinct relation types
              among edges with head entity `e`.
            - **h_rel_list** (list): List of unique relation types among edges
              with head entity `e`.
            - **t_unique_rel** (int): Number of distinct relation types
              among edges with tail entity `e`.
            - **t_rel_list** (list): List of unique relation types among edges
              with tail entity `e`.
            - **n_loops** (int): number of loops around entity `e`.
        """
        h_rel_list = {"h_rel_list": ("r", "unique")} if return_relation_list else {}
        t_rel_list = {"t_rel_list": ("r", "unique")} if return_relation_list else {}
        nodes = pd.DataFrame(
            self.df.groupby("h").agg(
                h_degree=("r", "count"), h_unique_rel=("r", "nunique"), **h_rel_list  # type: ignore
            ),
            index=np.arange(self.n_entity),
        )
        nodes = nodes.merge(
            self.df.groupby("t").agg(
                t_degree=("r", "count"), t_unique_rel=("r", "nunique"), **t_rel_list  # type: ignore
            ),
            left_index=True,
            right_index=True,
            how="left",
        )
        nodes = nodes.merge(
            self.df[self.df.h == self.df.t].groupby("h").agg(n_loops=("r", "count")),
            left_index=True,
            right_index=True,
            how="left",
        )
        nodes[["h_degree", "h_unique_rel", "t_degree", "t_unique_rel", "n_loops"]] = (
            nodes[["h_degree", "h_unique_rel", "t_degree", "t_unique_rel", "n_loops"]]
            .fillna(0)
            .astype(int)
        )
        nodes["tot_degree"] = nodes["h_degree"] + nodes["t_degree"] - nodes["n_loops"]

        return nodes[
            ["h_degree", "t_degree", "tot_degree", "h_unique_rel"]
            + (["h_rel_list"] if return_relation_list else [])
            + ["t_unique_rel"]
            + (["t_rel_list"] if return_relation_list else [])
            + ["n_loops"]
        ]

    def edge_degree_cardinality_summary(self) -> pd.DataFrame:
        """
        For each triple, this function computes the number of edges with the same head
        (head-degree, or out-degree), the same tail (tail-degree, or in-degree)
        or one of the two (total-degree) in the Knowledge Graph.
        Based on entity degrees, each triple is classified as either one-to-one
        (out-degree=in-degree=1), one-to-many (out-degree>1, in-degree=1),
        many-to-one(out-degree=1, in-degree>1) or many-to-many
        (in-degree>1, out-degree>1).

        The output dataframe maintains the same indexing and ordering of triples
        as the original Knowledge Graph dataframe.

        :return:
            The results dataframe. Contains the following columns
            (in addition to `h`, `r`, `t` in ``df``):

            - **h_unique_rel** (int): Number of distinct relation types
              among edges with head entity h.
            - **h_degree** (int): Number of triples with head entity h.
            - **h_degree_same_rel** (int): Number of triples with head entity h
              and relation type r.
            - **t_unique_rel** (int): Number of distinct relation types
              among edges with tail entity t.
            - **t_degree** (int): Number of triples with tail entity t.
            - **t_degree_same_rel** (int): Number of triples with tail entity t
              and relation type r.
            - **tot_degree** (int): Number of triples with head entity h or
              tail entity t.
            - **tot_degree_same_rel** (int): Number of triples with head entity h or
              tail entity t, and relation type r.
            - **triple_cardinality** (int): cardinality type of the edge.
            - **triple_cardinality_same_rel** (int): cardinality type of the edge in
              the subgraph of edges with relation type r.
        """
        gr_by_h_count = self.df.groupby("h", as_index=False).agg(
            h_unique_rel=("r", "nunique"), h_degree=("t", "count")
        )
        gr_by_hr_count = self.df.groupby(["h", "r"], as_index=False).agg(
            h_degree_same_rel=("t", "count")
        )
        gr_by_t_count = self.df.groupby("t", as_index=False).agg(
            t_unique_rel=("r", "nunique"), t_degree=("h", "count")
        )
        gr_by_rt_count = self.df.groupby(["r", "t"], as_index=False).agg(
            t_degree_same_rel=("h", "count")
        )

        df_res = self.df.merge(gr_by_h_count, left_on=["h"], right_on=["h"], how="left")
        df_res = df_res.merge(
            gr_by_hr_count, left_on=["h", "r"], right_on=["h", "r"], how="left"
        )
        df_res = df_res.merge(gr_by_t_count, left_on=["t"], right_on=["t"], how="left")
        df_res = df_res.merge(
            gr_by_rt_count, left_on=["t", "r"], right_on=["t", "r"], how="left"
        )
        # compute number of parallel edges to avoid double-counting them
        # in total degree
        num_parallel = df_res.merge(
            self.df.groupby(["h", "t"], as_index=False).agg(n_parallel=("r", "count")),
            left_on=["h", "t"],
            right_on=["h", "t"],
            how="left",
        )
        df_res["tot_degree"] = (
            df_res.h_degree + df_res.t_degree - num_parallel.n_parallel
        )
        # when restricting to the relation type, there is only one edge
        # (the edge itself) that is double-counted
        df_res["tot_degree_same_rel"] = (
            df_res.h_degree_same_rel + df_res.t_degree_same_rel - 1
        )

        # check if the values in the pair (h_degree, t_degree) are =1 or >1
        # to determine the edge cardinality
        legend = {
            0: "M:M",
            1: "1:M",
            2: "M:1",
            3: "1:1",
        }
        for suffix in ["", "_same_rel"]:
            edge_type = 2 * (df_res["h_degree" + suffix] == 1) + (
                df_res["t_degree" + suffix] == 1
            )
            df_res["triple_cardinality" + suffix] = edge_type.apply(lambda x: legend[x])
        return df_res

    def edge_pattern_summary(
        self,
        return_metapath_list: bool = False,
        composition_chunk_size: int = 2**8,
        composition_workers: int = 32,
    ) -> pd.DataFrame:
        """
        This function analyses the structural properties of each edge in the graph:
        symmetry, presence of inverse/inference(=parallel) edges and
        triangles supported on the edge.

        The output dataframe maintains the same indexing and ordering of triples
        as the original Knowledge Graph dataframe.

        :param return_metapath_list: If True, return the list of unique metapaths for all
            triangles supported over one edge. WARNING: very expensive for large graphs.
        :param composition_chunk_size: Size of column chunks of sparse adjacency matrix
            to compute the triangle count.
        :param composition_workers: Number of workers to compute the triangle count.

        :return: The results dataframe. Contains the following columns
            (in addition to `h`, `r`, `t` in ``df``):

            - **is_loop** (bool): True if the triple is a loop (``h == t``).
            - **is_symmetric** (bool): True if the triple (t, r, h) is also contained
              in the graph (assuming t and h are different).
            - **has_inverse** (bool): True if the graph contains one or more triples
              (t, r', h) with ``r' != r``.
            - **n_inverse_relations** (int): The number of inverse relations r'.
            - **inverse_edge_types** (list): All relations r' (including r if the edge
              is symmetric) such that (t, r', h) is in the graph.
            - **has_inference** (bool): True if the graph contains one or more triples
              (h, r', t) with ``r' != r``.
            - **n_inference_relations** (int): The number of inference relations r'.
            - **inference_edge_types** (list): All relations r' (including r) such that
              (h, r', t) is in the graph.
            - **has_composition** (bool): True if the graph contains one or more triangles
              supported on the edge: (h, r1, x) + (x, r2, t).
            - **n_triangles** (int): The number of triangles.
            - **has_undirected_composition** (bool): True if the graph contains one or more
              undirected triangles supported on the edge.
            - **n_undirected_triangles** (int): The number of undirected triangles
              (considering all edges as bidirectional).
            - **metapath_list** (list): The list of unique metapaths "r1-r2"
              for the directed triangles.
        """
        # symmetry-asymmetry
        # edges with h/t switched
        df_inv = self.df.reindex(columns=["t", "r", "h"]).rename(
            columns={"t": "h", "r": "r", "h": "t"}
        )
        df_res = pd.DataFrame(
            {"h": self.df.h, "r": self.df.r, "t": self.df.t, "is_symmetric": False}
        )
        df_res.loc[
            self.df.reset_index().merge(df_inv)["index"],
            "is_symmetric",
        ] = True
        # loops are treated separately
        df_res["is_loop"] = df_res.h == df_res.t
        df_res.loc[df_res.h == df_res.t, "is_symmetric"] = False

        # inverse
        unique_inv_r_by_ht = df_inv.groupby(["h", "t"], as_index=False).agg(
            inverse_edge_types=("r", list),
        )
        df_res = df_res.merge(
            unique_inv_r_by_ht, left_on=["h", "t"], right_on=["h", "t"], how="left"
        )
        df_res["inverse_edge_types"] = df_res["inverse_edge_types"].apply(
            lambda agg: agg if isinstance(agg, list) else []
        )
        # if the edge (h,r,t) is symmetric or loop, we do not consider the relation
        # r as a proper inverse
        df_res["n_inverse_relations"] = (
            df_res.inverse_edge_types.str.len() - df_res.is_symmetric - df_res.is_loop
        )
        df_res["n_inverse_relations"] = (
            df_res["n_inverse_relations"].fillna(0).astype(int)
        )
        df_res["has_inverse"] = df_res["n_inverse_relations"] > 0

        # inference
        edges_between_ht = unique_inv_r_by_ht.reindex(
            columns=["t", "h", "inverse_edge_types"]
        ).rename(
            columns={"t": "h", "h": "t", "inverse_edge_types": "inference_edge_types"}
        )
        df_res = df_res.merge(
            edges_between_ht, left_on=["h", "t"], right_on=["h", "t"], how="left"
        )
        # inference_edge_types always contains the edge itself, which we need to drop
        df_res["n_inference_relations"] = df_res.inference_edge_types.str.len() - 1
        df_res["has_inference"] = df_res["n_inference_relations"] > 0

        # composition & metapaths
        # discard loops as edges of a triangle
        df_wo_loops = self.df[self.df.h != self.df.t]
        if return_metapath_list:
            # 2-hop paths
            df_bridges = df_wo_loops.merge(
                df_wo_loops, left_on="t", right_on="h", how="inner"
            )
            df_triangles = df_wo_loops.merge(
                df_bridges, left_on=["h", "t"], right_on=["h_x", "t_y"], how="inner"
            )
            df_triangles["metapath"] = (
                df_triangles["r_x"].astype(str) + "-" + df_triangles["r_y"].astype(str)
            )
            grouped_triangles = df_triangles.groupby(
                ["h", "r", "t"], as_index=False
            ).agg(
                n_triangles=("metapath", "count"), metapath_list=("metapath", "unique")
            )
            df_res = df_res.merge(
                grouped_triangles,
                left_on=["h", "r", "t"],
                right_on=["h", "r", "t"],
                how="left",
            )
            df_res["metapath_list"] = df_res["metapath_list"].apply(
                lambda agg: agg.tolist() if isinstance(agg, np.ndarray) else []
            )
            df_res["n_triangles"] = df_res["n_triangles"].fillna(0).astype(int)
        else:
            counts = composition_count(
                df_wo_loops,
                chunk_size=composition_chunk_size,
                workers=composition_workers,
                directed=True,
            )
            df_res = df_res.merge(
                counts,
                on=["h", "t"],
                how="left",
            )
            df_res["n_triangles"] = df_res["n_triangles"].fillna(0).astype(int)

        df_res["has_composition"] = df_res["n_triangles"] > 0

        counts = composition_count(
            df_wo_loops,
            chunk_size=composition_chunk_size,
            workers=composition_workers,
            directed=False,
        )
        df_res = df_res.merge(
            counts.rename(columns={"n_triangles": "n_undirected_triangles"}),
            on=["h", "t"],
            how="left",
        )
        df_res["n_undirected_triangles"] = (
            df_res["n_undirected_triangles"].fillna(0).astype(int)
        )
        df_res["has_undirected_composition"] = df_res["n_undirected_triangles"] > 0

        return df_res[
            [
                "h",
                "r",
                "t",
                "is_loop",
                "is_symmetric",
                "has_inverse",
                "n_inverse_relations",
                "inverse_edge_types",
                "has_inference",
                "n_inference_relations",
                "inference_edge_types",
                "has_composition",
                "has_undirected_composition",
                "n_triangles",
                "n_undirected_triangles",
            ]
            + (["metapath_list"] if return_metapath_list else [])
        ]

    def aggregate_by_relation(self, edge_topology_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate topology metrics of all triples of the same relation type.
        To be applied to the output dataframe of either
        :meth:`KGTopologyToolbox.edge_degree_cardinality_summary` or
        :meth:`KGTopologyToolbox.edge_pattern_summary`.

        The returned dataframe is indexed over relation type IDs, with columns
        giving the aggregated statistics of triples of the correspondig relation.
        The name of the columns is of the form ``column_name_in_input_df + suffix``.
        The aggregation is perfomed by returning:

        - for numerical metrics: mean, standard deviation and quartiles
          (``suffix`` = "_mean", "_std", "_quartile1", "_quartile2", "_quartile3");
        - for boolean metrics: the fraction of triples of the relation type
          with metric = True (``suffix`` = "_frac");
        - for string metrics: for each possible label, the fraction of triples
          of the relation type with that metric value (``suffix`` = "_{label}_frac")
        - for list metrics: the unique metric values across triples of the relation
          type (``suffix`` = "_unique").

        :param edge_topology_df: pd.DataFrame of edge topology metrics.
            Must contain at least three columns `h`, `r`, `t`.

        :return: The results dataframe. In addition to the columns with the aggregated
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

    def jaccard_similarity_relation_sets(self) -> pd.DataFrame:
        """
        Compute the similarity between relations defined as the Jaccard Similarity
        between sets of entities (heads and tails) for all pairs
        of relations in the graph.

        :return:
            The results dataframe. Contains the following columns:

            - **r1** (int): Index of the first relation.
            - **r2** (int): Index of the second relation.
            - **num_triples_both** (int): Number of triples with relation r1/r2.
            - **frac_triples_both** (float): Fraction of triples with relation r1/r2.
            - **num_entities_both** (int): Number of unique entities (h or t) for triples
              with relation r1/r2.
            - **num_h_r1** (int): Number of unique head entities for relation r1.
            - **num_h_r2** (int): Number of unique head entities for relation r2.
            - **num_t_r1** (int): Number of unique tail entities for relation r1.
            - **num_t_r2** (int): Number of unique tail entities for relation r2.
            - **jaccard_head_head** (float): Jaccard similarity between the head set of r1
              and the head set of r2.
            - **jaccard_tail_tail** (float): Jaccard similarity between the tail set of r1
              and the tail set of r2.
            - **jaccard_head_tail** (float): Jaccard similarity between the head set of r1
              and the tail set of r2.
            - **jaccard_tail_head** (float): Jaccard similarity between the tail set of r1
              and the head set of r2.
            - **jaccard_both** (float): Jaccard similarity between the full entity set
              of r1 and r2.
        """
        ent_unique = self.df.groupby("r", as_index=False).agg(
            num_triples=("r", "count"), head=("h", "unique"), tail=("t", "unique")
        )
        ent_unique["both"] = ent_unique.apply(
            lambda x: np.unique(np.concatenate([x["head"], x["tail"]])), axis=1
        )
        ent_unique["num_h"] = ent_unique["head"].str.len()
        ent_unique["num_t"] = ent_unique["tail"].str.len()
        r_num = ent_unique[["r", "num_h", "num_t", "num_triples"]]
        # combinations of relations
        df_res = pd.merge(
            r_num.rename(columns={"r": "r1"}),
            r_num.rename(columns={"r": "r2"}),
            suffixes=["_r1", "_r2"],
            how="cross",
        )
        df_res = df_res[df_res.r1 < df_res.r2]

        df_res["num_triples_both"] = df_res["num_triples_r1"] + df_res["num_triples_r2"]
        df_res["frac_triples_both"] = df_res["num_triples_both"] / self.df.shape[0]
        df_res["num_entities_both"] = df_res.apply(
            lambda x: len(
                np.unique(
                    np.concatenate(
                        [
                            ent_unique.loc[x["r1"], "both"],
                            ent_unique.loc[x["r2"], "both"],
                        ]
                    )
                )
            ),
            axis=1,
        )
        df_res = df_res[
            [
                "r1",
                "r2",
                "num_triples_both",
                "frac_triples_both",
                "num_entities_both",
                "num_h_r1",
                "num_h_r2",
                "num_t_r1",
                "num_t_r2",
            ]
        ]
        for r1_ent in ["head", "tail"]:
            for r2_ent in ["head", "tail"]:
                df_res[f"jaccard_{r1_ent}_{r2_ent}"] = [
                    jaccard_similarity(a, b)
                    for a, b in zip(
                        ent_unique.loc[df_res.r1, r1_ent],
                        ent_unique.loc[df_res.r2, r2_ent],
                    )
                ]
        df_res["jaccard_both"] = [
            jaccard_similarity(a, b)
            for a, b in zip(
                ent_unique.loc[df_res.r1, "both"], ent_unique.loc[df_res.r2, "both"]
            )
        ]
        return df_res

    def relational_affinity_ingram(self, min_max_norm: bool = False) -> pd.DataFrame:
        """
        Compute the similarity between relations based on the approach proposed in
        InGram: Inductive Knowledge Graph Embedding via Relation Graphs,
        https://arxiv.org/abs/2305.19987.

        Only the pairs of relations witn ``affinity > 0`` are shown in the
        returned dataframe.

        :param min_max_norm:
            min-max normalization of edge weights. Defaults to False.

        :return:
            The results dataframe. Contains the following columns:

            - **h_relation** (int): Index of the head relation.
            - **t_relation** (int): Index of the tail relation.
            - **edge_weight** (float): Weight for the affinity between
              the head and the tail relation.
        """
        hr_freqs = self.df.groupby(["h", "r"], as_index=False).count()
        # normalize by global h frequency
        hr_freqs["t"] = hr_freqs["t"] / hr_freqs.groupby("h")["t"].transform("sum")
        rt_freqs = self.df.groupby(["t", "r"], as_index=False).count()
        # normalize by global t frequency
        rt_freqs["h"] = rt_freqs["h"] / rt_freqs.groupby("t")["h"].transform("sum")

        E_h = coo_array(
            (hr_freqs.t, (hr_freqs.h, hr_freqs.r)),
            shape=[self.n_entity, self.n_rel],
        )
        E_t = coo_array(
            (rt_freqs.h, (rt_freqs.t, rt_freqs.r)),
            shape=[self.n_entity, self.n_rel],
        )

        A = (E_h.T @ E_h).toarray() + (E_t.T @ E_t).toarray()
        A[np.diag_indices_from(A)] = 0

        if min_max_norm:
            A = (A - np.min(A)) / (np.max(A) - np.min(A))

        h_rels, t_rels = np.nonzero(A)
        return pd.DataFrame(
            {
                "h_relation": h_rels,
                "t_relation": t_rels,
                "edge_weight": A[h_rels, t_rels],
            }
        )
