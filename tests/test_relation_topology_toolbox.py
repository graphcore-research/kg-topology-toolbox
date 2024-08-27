# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import List

import numpy as np
import pandas as pd
import pytest

from kg_topology_toolbox import KGTopologyToolbox

df = pd.DataFrame(
    dict(
        H=[0, 0, 0, 1, 2, 2, 2, 3, 3, 4],
        T=[1, 1, 2, 2, 0, 3, 4, 2, 4, 3],
        R=[0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        n=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )
)

kgtt = KGTopologyToolbox(df, head_column="H", relation_column="R", tail_column="T")


def test_small_graph_metrics() -> None:
    # Define a small graph on five nodes with all the features tested by
    # the relation_topology_toolbox

    dcs = kgtt.edge_degree_cardinality_summary(aggregate_by_r=True)
    eps = kgtt.edge_pattern_summary(return_metapath_list=True, aggregate_by_r=True)

    assert np.allclose(dcs["num_triples"], [5, 5])
    assert np.allclose(dcs["frac_triples"], [0.5, 0.5])
    assert np.allclose(dcs["unique_h"], [3, 4])
    assert np.allclose(dcs["unique_t"], [4, 4])

    # entity_degree_statistics
    assert np.allclose(dcs["h_degree_mean"], [2.6, 2.2])
    assert np.allclose(dcs["t_degree_mean"], [2.2, 2.2])
    assert np.allclose(dcs["tot_degree_mean"], [3.6, 3.2])

    # triple_relation_cardinality
    assert np.allclose(dcs["triple_cardinality_1:M_frac"], [1 / 5, 0])
    assert np.allclose(dcs["triple_cardinality_M:1_frac"], [0, 2 / 5])
    assert np.allclose(dcs["triple_cardinality_M:M_frac"], [4 / 5, 3 / 5])
    assert np.allclose(dcs["triple_cardinality_same_rel_1:1_frac"], [1 / 5, 2 / 5])
    assert np.allclose(dcs["triple_cardinality_same_rel_1:M_frac"], [2 / 5, 1 / 5])
    assert np.allclose(dcs["triple_cardinality_same_rel_M:1_frac"], [0, 1 / 5])
    assert np.allclose(dcs["triple_cardinality_same_rel_M:M_frac"], [2 / 5, 1 / 5])

    # relation_pattern_loop
    assert np.allclose(eps["is_loop_frac"], [0, 0])

    # relation_pattern_symmetric
    assert np.allclose(eps["is_symmetric_frac"], [2 / 5, 0])

    # relation_pattern_inverse
    assert np.allclose(eps["has_inverse_frac"], [2 / 5, 2 / 5])
    assert eps["inverse_edge_types_unique"][0] == [0, 1]
    assert eps["inverse_edge_types_unique"][1] == [0]

    # relation_pattern_composition
    assert np.allclose(eps["has_composition_frac"], [2 / 5, 2 / 5])
    assert np.allclose(eps["has_undirected_composition_frac"], [1, 1])
    assert eps["metapath_list_unique"][0] == ["0-1", "1-1"]
    assert eps["metapath_list_unique"][1] == ["1-0", "1-1"]

    # relation_pattern_inference
    assert np.allclose(eps["has_inference_frac"], [1 / 5, 1 / 5])
    assert eps["inference_edge_types_unique"][0] == [0, 1]
    assert eps["inference_edge_types_unique"][1] == [0, 1]


def test_jaccard_similarity() -> None:
    # jaccard_similarity_relation_sets
    res = kgtt.jaccard_similarity_relation_sets()
    assert np.allclose(res["jaccard_head_head"], [2 / 5])
    assert np.allclose(res["jaccard_tail_tail"], [3 / 5])
    assert np.allclose(res["jaccard_head_tail"], [2 / 5])
    assert np.allclose(res["jaccard_tail_head"], [1])
    assert np.allclose(res["jaccard_both"], [1])


@pytest.mark.parametrize(
    "min_max_norm,expected", [(True, [1, 1]), (False, [7 / 6, 7 / 6])]
)
def test_ingram_affinity(min_max_norm: bool, expected: List[float]) -> None:
    # relational_affinity_ingram
    res = kgtt.relational_affinity_ingram(min_max_norm)
    assert np.allclose(res["edge_weight"], expected)
