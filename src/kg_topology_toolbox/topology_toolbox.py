# -*- coding: utf-8 -*-
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Topology toolbox main functionalities
"""

import multiprocessing as mp
from functools import cache

import numpy as np
import pandas as pd
from scipy.sparse import coo_array

from kg_topology_toolbox.utils import (
    aggregate_by_relation,
    check_kg_df_structure,
    composition_count,
    jaccard_similarity,
    node_degrees_and_rels,
)


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
        :param relation_column:
            The name of the column with the IDs of relation types. Default: "r".
        :param tail_column:
            The name of the column with the IDs of tail entities. Default: "t".

        """
        check_kg_df_structure(kg_df, head_column, relation_column, tail_column)

        self.df = kg_df[[head_column, relation_column, tail_column]].rename(
            columns={head_column: "h", relation_column: "r", tail_column: "t"}
        )
        self.n_entity = self.df[["h", "t"]].max().max() + 1
        self.n_rel = self.df.r.max() + 1

    def loop_count(self) -> pd.DataFrame:
        """
        For each entity in the KG, compute the number of loops around the entity
        (i.e., the number of edges having the entity as both head and tail).

        :return:
            Loop count DataFrame, indexed on the IDs of the graph entities.
        """
        n_loops = (
            self.df[self.df.h == self.df.t].groupby("h").agg(n_loops=("r", "count"))
        )
        return (
            pd.DataFrame(n_loops, index=np.arange(self.n_entity)).fillna(0).astype(int)
        )

    @cache
    def node_head_degree(self, return_relation_list: bool = False) -> pd.DataFrame:
        """
        For each entity in the KG, compute the number of edges having it as head
        (head-degree, or out-degree of the head node).
        The relation types going out of the head node are also identified.

        :param return_relation_list:
            If True, return the list of unique relations going
            out of the head node. WARNING: expensive for large graphs.
            Default: False.

        :return:
            The result DataFrame, indexed on the IDs `e` of the graph entities,
            with columns:

            - **h_degree** (int): Number of triples with head entity `e`.
            - **h_unique_rel** (int): Number of distinct relation types
              among edges with head entity `e`.
            - **h_rel_list** (Optional[list]): List of unique relation types
              among edges with head entity `e`.
              Only returned if `return_relation_list = True`.
        """
        node_df = node_degrees_and_rels(
            self.df, "h", self.n_entity, return_relation_list
        )
        return node_df.rename(columns={n: "h_" + n for n in node_df.columns})

    @cache
    def node_tail_degree(self, return_relation_list: bool = False) -> pd.DataFrame:
        """
        For each entity in the KG, compute the number of edges having it as tail
        (tail-degree, or in-degree of the tail node).
        The relation types going into the tail node are also identified.

        :param return_relation_list:
            If True, return the list of unique relation types going
            into the tail node. WARNING: expensive for large graphs.
            Default: False.

        :return:
            The result DataFrame, indexed on the IDs `e` of the graph entities,
            with columns:

            - **t_degree** (int): Number of triples with tail entity `e`.
            - **t_unique_rel** (int): Number of distinct relation types
              among edges with tail entity `e`.
            - **t_rel_list** (Optional[list]): List of unique relation types
              among edges with tail entity `e`.
              Only returned if `return_relation_list = True`.
        """
        node_df = node_degrees_and_rels(
            self.df, "t", self.n_entity, return_relation_list
        )
        return node_df.rename(columns={n: "t_" + n for n in node_df.columns})

    def node_degree_summary(self, return_relation_list: bool = False) -> pd.DataFrame:
        """
        For each entity in the KG, compute the number of edges having it as a head
        (head-degree, or out-degree), as a tail (tail-degree, or in-degree)
        or one of the two (total-degree).
        The in-going and out-going relation types are also identified.

        The output dataframe is indexed on the IDs of the graph entities.

        :param return_relation_list:
            If True, return the list of unique relations going
            in/out of an entity. WARNING: expensive for large graphs.

        :return:
            The results dataframe, indexed on the IDs `e` of the graph entities,
            with columns:

            - **h_degree** (int): Number of triples with head entity `e`.
            - **t_degree** (int): Number of triples with tail entity `e`.
            - **tot_degree** (int): Number of triples with head entity `e` or tail entity `e`.
            - **h_unique_rel** (int): Number of distinct relation types
              among edges with head entity `e`.
            - **h_rel_list** (Optional[list]): List of unique relation types among edges
              with head entity `e`.
              Only returned if `return_relation_list = True`.
            - **t_unique_rel** (int): Number of distinct relation types
              among edges with tail entity `e`.
            - **t_rel_list** (Optional[list]): List of unique relation types among edges
              with tail entity `e`.
              Only returned if `return_relation_list = True`.
            - **n_loops** (int): number of loops around entity `e`.
        """
        nodes_df = pd.merge(
            self.node_head_degree(return_relation_list),
            self.node_tail_degree(return_relation_list),
            left_index=True,
            right_index=True,
        )
        nodes_df = pd.merge(
            nodes_df,
            self.loop_count(),
            left_index=True,
            right_index=True,
        )
        nodes_df["tot_degree"] = (
            nodes_df["h_degree"] + nodes_df["t_degree"] - nodes_df["n_loops"]
        )

        return nodes_df[
            ["h_degree", "t_degree", "tot_degree", "h_unique_rel"]
            + (["h_rel_list"] if return_relation_list else [])
            + ["t_unique_rel"]
            + (["t_rel_list"] if return_relation_list else [])
            + ["n_loops"]
        ]

    @cache
    def edge_head_degree(self) -> pd.DataFrame:
        """
        For each edge in the KG, compute the number of edges
        (in total or of the same relation type) with the same head node.

        :return:
            The result DataFrame, with the same indexing and ordering of
            triples as the original KG DataFrame, with columns
            (in addition to `h`, `r`, `t`):

            - **h_unique_rel** (int): Number of distinct relation types
              among edges with head entity `h`.
            - **h_degree** (int): Number of triples with head entity `h`.
            - **h_degree_same_rel** (int): Number of triples with head entity `h`
              and relation type `r`.
        """
        edge_by_hr_count = self.df.groupby(["h", "r"], as_index=False).agg(
            h_degree_same_rel=("t", "count")
        )
        df_res = self.df.merge(
            self.node_head_degree(), left_on=["h"], right_index=True, how="left"
        )
        return df_res.merge(edge_by_hr_count, on=["h", "r"], how="left")

    @cache
    def edge_tail_degree(self) -> pd.DataFrame:
        """
        For each edge in the KG, compute the number of edges
        (in total or of the same relation type) with the same tail node.

        :return:
            The result DataFrame, with the same indexing and ordering of
            triples as the original KG DataFrame, with columns
            (in addition to `h`, `r`, `t`):

            - **t_unique_rel** (int): Number of distinct relation types
              among edges with tail entity `t`.
            - **t_degree** (int): Number of triples with tail entity `t`.
            - **t_degree_same_rel** (int): Number of triples with tail entity `t`
              and relation type `r`.
        """
        edge_by_rt_count = self.df.groupby(["r", "t"], as_index=False).agg(
            t_degree_same_rel=("h", "count")
        )
        df_res = self.df.merge(
            self.node_tail_degree(), left_on=["t"], right_index=True, how="left"
        )
        return df_res.merge(edge_by_rt_count, on=["r", "t"], how="left")

    def edge_cardinality(self) -> pd.DataFrame:
        """
        Classify the cardinality of each edge in the KG: one-to-one
        (out-degree=in-degree=1), one-to-many (out-degree>1, in-degree=1),
        many-to-one(out-degree=1, in-degree>1) or many-to-many
        (in-degree>1, out-degree>1).

        :return:
            The result DataFrame, with the same indexing and ordering of
            triples as the original KG DataFrame, with columns
            (in addition to `h`, `r`, `t`):

            - **triple_cardinality** (int): cardinality type of the edge.
            - **triple_cardinality_same_rel** (int): cardinality type of the edge in
              the subgraph of edges with relation type `r`.
        """
        head_degree = self.edge_head_degree()
        tail_degree = self.edge_tail_degree()
        df_res = pd.DataFrame(
            {"h": head_degree.h, "r": head_degree.r, "t": head_degree.t}
        )
        # check if the values in the pair (h_degree, t_degree) are =1 or >1
        # to determine the edge cardinality
        for suffix in ["", "_same_rel"]:
            edge_type = 2 * (head_degree["h_degree" + suffix] == 1) + (
                tail_degree["t_degree" + suffix] == 1
            )
            df_res["triple_cardinality" + suffix] = pd.cut(
                edge_type,
                bins=[0, 1, 2, 3, 4],
                right=False,
                labels=["M:M", "1:M", "M:1", "1:1"],
            ).astype(str)
        return df_res

    def edge_metapath_count(
        self,
        filter_relations: list[int] = [],
        composition_chunk_size: int = 2**8,
        composition_workers: int = min(32, mp.cpu_count() - 1 or 1),
    ) -> pd.DataFrame:
        """
        For each edge in the KG, compute the number of triangles supported on it
        distinguishing between different metapaths (i.e., the unique ordered tuples
        (r1, r2) of relation types of the two additional edges of the triangle).

        :param filter_relations:
            If not empty, compute the output only for the edges with relation
            in this list of relation IDs.
        :param composition_chunk_size:
            Size of column chunks of sparse adjacency matrix
            to compute the triangle count. Reduce the parameter if running OOM.
            Default: 2**8.
        :param composition_workers:
            Number of workers to compute the triangle count. By default, assigned based
            on number of available threads (max: 32).

        :return:
            The output dataframe has one row for each (h, r, t, r1, r2) such that
            there exists at least one triangle of metapath (r1, r2) over (h, r, t).
            The number of metapath triangles is given in the column **n_triangles**.
            The column **index** provides the index of the edge (h, r, t) in the
            original Knowledge Graph dataframe.
        """
        # discard loops as edges of a triangle
        df_wo_loops = self.df[self.df.h != self.df.t]
        if len(filter_relations) > 0:
            rel_df = self.df[self.df.r.isin(filter_relations)]
            # unique heads and tails used by filtered edges
            filter_heads = rel_df.h.unique()
            filter_tails = rel_df.t.unique()
            # the only relevant edges for triangles are the ones with head in the
            # set of filtered heads, or tail in the set of filtered tails
            df_triangles = df_wo_loops[
                np.logical_or(
                    df_wo_loops.h.isin(filter_heads), df_wo_loops.t.isin(filter_tails)
                )
            ]
        else:
            rel_df = self.df
            df_triangles = df_wo_loops

        counts = composition_count(
            df_triangles,
            chunk_size=composition_chunk_size,
            workers=composition_workers,
            metapaths=True,
            directed=True,
        )

        return rel_df.reset_index().merge(counts, on=["h", "t"], how="inner")

    def edge_degree_cardinality_summary(
        self, filter_relations: list[int] = [], aggregate_by_r: bool = False
    ) -> pd.DataFrame:
        """
        For each edge in the KG, compute the number of edges with the same head
        (head-degree, or out-degree), the same tail (tail-degree, or in-degree)
        or one of the two (total-degree).
        Based on entity degrees, each triple is classified as either one-to-one
        (out-degree=in-degree=1), one-to-many (out-degree>1, in-degree=1),
        many-to-one(out-degree=1, in-degree>1) or many-to-many
        (in-degree>1, out-degree>1).

        The output dataframe maintains the same indexing and ordering of triples
        as the original Knowledge Graph dataframe.

        :param filter_relations:
            If not empty, compute the output only for the edges with relation
            in this list of relation IDs.
        :param aggregate_by_r:
            If True, return metrics aggregated by relation type
            (the output DataFrame will be indexed over relation IDs).

        :return:
            The results dataframe. Contains the following columns
            (in addition to `h`, `r`, `t`):

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
        df_res = pd.concat(
            [
                self.edge_head_degree(),
                self.edge_tail_degree().drop(columns=["h", "r", "t"]),
            ],
            axis=1,
        )
        if len(filter_relations) > 0:
            df_res = df_res[df_res.r.isin(filter_relations)]
        # compute number of parallel edges to avoid double-counting them
        # in total degree
        num_parallel = df_res.merge(
            self.df.groupby(["h", "t"], as_index=False).agg(n_parallel=("r", "count")),
            on=["h", "t"],
            how="left",
        )
        df_res["tot_degree"] = (
            df_res.h_degree + df_res.t_degree - num_parallel.n_parallel.values
        )
        # when restricting to the same relation type, there is only one edge
        # (the edge itself) that is double-counted
        df_res["tot_degree_same_rel"] = (
            df_res.h_degree_same_rel + df_res.t_degree_same_rel - 1
        )

        edge_cardinality = self.edge_cardinality()
        df_res["triple_cardinality"] = edge_cardinality["triple_cardinality"]
        df_res["triple_cardinality_same_rel"] = edge_cardinality[
            "triple_cardinality_same_rel"
        ]
        return aggregate_by_relation(df_res) if aggregate_by_r else df_res

    def edge_pattern_summary(
        self,
        return_metapath_list: bool = False,
        filter_relations: list[int] = [],
        aggregate_by_r: bool = False,
        composition_chunk_size: int = 2**8,
        composition_workers: int = min(32, mp.cpu_count() - 1 or 1),
    ) -> pd.DataFrame:
        """
        Analyse structural properties of each edge in the KG:
        symmetry, presence of inverse/inference(=parallel) edges and
        triangles supported on the edge.

        The output dataframe maintains the same indexing and ordering of triples
        as the original Knowledge Graph dataframe.

        :param return_metapath_list:
            If True, return the list of unique metapaths for all
            triangles supported over each edge. WARNING: very expensive for large graphs.
        :param filter_relations:
            If not empty, compute the output only for the edges with relation
            in this list of relation IDs.
        :param aggregate_by_r:
            If True, return metrics aggregated by relation type
            (the output DataFrame will be indexed over relation IDs).
        :param composition_chunk_size:
            Size of column chunks of sparse adjacency matrix
            to compute the triangle count. Reduce the parameter if running OOM.
            Default: 2**8.
        :param composition_workers:
            Number of workers to compute the triangle count. By default, assigned based
            on number of available threads (max: 32).

        :return:
            The results dataframe. Contains the following columns
            (in addition to `h`, `r`, `t`):

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

        # discard loops as edges of a triangle
        df_wo_loops = self.df[self.df.h != self.df.t]
        if len(filter_relations) > 0:
            rel_df = self.df[self.df.r.isin(filter_relations)]
            # unique heads and tails used by filtered edges
            filter_heads = rel_df.h.unique()
            filter_tails = rel_df.t.unique()
            filter_entities = np.union1d(filter_heads, filter_tails)
            # restrict relevant edges to count inference/inverse patterns
            inference_df = self.df[
                np.logical_and(
                    self.df.h.isin(filter_heads), self.df.t.isin(filter_tails)
                )
            ]
            inverse_df = self.df[
                np.logical_and(
                    self.df.h.isin(filter_tails), self.df.t.isin(filter_heads)
                )
            ]
            # the only relevant edges for triangles are the ones with head in the
            # set of filtered heads, or tail in the set of filtered tails
            df_triangles = df_wo_loops[
                np.logical_or(
                    df_wo_loops.h.isin(filter_heads), df_wo_loops.t.isin(filter_tails)
                )
            ]
            # for undirected triangles, heads and tails can be any of the
            # filtered entities
            df_triangles_und = df_wo_loops[
                np.logical_or(
                    df_wo_loops.h.isin(filter_entities),
                    df_wo_loops.t.isin(filter_entities),
                )
            ]
        else:
            rel_df = inference_df = inverse_df = self.df
            df_triangles = df_triangles_und = df_wo_loops
        df_res = pd.DataFrame(
            {"h": rel_df.h, "r": rel_df.r, "t": rel_df.t, "is_symmetric": False}
        )
        # symmetry-asymmetry
        # edges with h/t switched
        df_inv = inverse_df.reindex(columns=["t", "r", "h"]).rename(
            columns={"t": "h", "r": "r", "h": "t"}
        )
        df_res.loc[
            df_res.reset_index().merge(df_inv)["index"],
            "is_symmetric",
        ] = True
        # loops are treated separately
        df_res["is_loop"] = df_res.h == df_res.t
        df_res.loc[df_res.h == df_res.t, "is_symmetric"] = False

        df_res = df_res.reset_index()

        # inverse
        unique_inv_r_by_ht = df_inv.groupby(["h", "t"], as_index=False).agg(
            inverse_edge_types=("r", list),
        )
        df_res = df_res.merge(unique_inv_r_by_ht, on=["h", "t"], how="left")
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
        if len(filter_relations) > 0:
            edges_between_ht = inference_df.groupby(["h", "t"], as_index=False).agg(
                inference_edge_types=("r", list),
            )
        else:
            edges_between_ht = unique_inv_r_by_ht.reindex(
                columns=["t", "h", "inverse_edge_types"]
            ).rename(
                columns={
                    "t": "h",
                    "h": "t",
                    "inverse_edge_types": "inference_edge_types",
                }
            )
        df_res = df_res.merge(edges_between_ht, on=["h", "t"], how="left")
        # inference_edge_types always contains the edge itself, which we need to drop
        df_res["n_inference_relations"] = df_res.inference_edge_types.str.len() - 1
        df_res["has_inference"] = df_res["n_inference_relations"] > 0

        # composition & metapaths
        counts = composition_count(
            df_triangles,
            chunk_size=composition_chunk_size,
            workers=composition_workers,
            metapaths=return_metapath_list,
            directed=True,
        )
        if return_metapath_list:
            # turn (r1, r2) into "r1-r2" string for metapaths
            counts["metapath"] = (
                counts["r1"].astype(str) + "-" + counts["r2"].astype(str)
            )
            # count triangles (summing over all metapaths between two nodes)
            # and list unique metapaths for each head and tail node pair
            grouped_triangles = counts.groupby(["h", "t"], as_index=False).agg(
                n_triangles=("n_triangles", "sum"), metapath_list=("metapath", list)
            )
            df_res = df_res.merge(
                grouped_triangles,
                on=["h", "t"],
                how="left",
            )
            # if no triangles are present over an edge, set metapath list to []
            df_res["metapath_list"] = df_res["metapath_list"].apply(
                lambda agg: agg if isinstance(agg, list) else []
            )
        else:
            df_res = df_res.merge(
                counts,
                on=["h", "t"],
                how="left",
            )
        df_res["n_triangles"] = df_res["n_triangles"].fillna(0).astype(int)
        df_res["has_composition"] = df_res["n_triangles"] > 0

        # undirected composition
        counts = composition_count(
            df_triangles_und,
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

        df_res = df_res.set_index("index")[
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
        df_res.index.name = None

        return aggregate_by_relation(df_res) if aggregate_by_r else df_res

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
        # set of unique heads/tails/any for each relation
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
        # order doesn't matter
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
            min-max normalization of edge weights. Default: False.

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

        # sparse matrix of of (h,r) pair frequency
        E_h = coo_array(
            (hr_freqs.t, (hr_freqs.h, hr_freqs.r)),
            shape=[self.n_entity, self.n_rel],
        )
        # sparse matrix of of (t,r) pair frequency
        E_t = coo_array(
            (rt_freqs.h, (rt_freqs.t, rt_freqs.r)),
            shape=[self.n_entity, self.n_rel],
        )

        # adjacency matrix of relation graph
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
