# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from functools import partial
import numpy as np
import pandas as pd
import pytest

from kg_topology_toolbox import KGTopologyToolbox

df = pd.DataFrame(
    dict(
        H=[0, 0, 0, 1, 2, 2, 1, 2],
        T=[1, 1, 2, 2, 0, 0, 1, 2],
        R=[0, 1, 0, 1, 0, 1, 1, 0],
        n=["a", "b", "c", "d", "e", "f", "g", "h"],
    )
)

kgtt = KGTopologyToolbox(
    kg_df=df, head_column="H", relation_column="R", tail_column="T"
)


@pytest.mark.parametrize("return_metapath_list", [True, False])
def test_small_graph_metrics(return_metapath_list: bool) -> None:
    # Define a small graph with all the features tested by
    # the edge_topology_toolbox

    # entity degrees statistics
    res = kgtt.edge_degree_cardinality_summary()
    assert np.allclose(res["h_unique_rel"], [2, 2, 2, 1, 2, 2, 1, 2])
    assert np.allclose(res["h_degree"], [3, 3, 3, 2, 3, 3, 2, 3])
    assert np.allclose(res["h_degree_same_rel"], [2, 1, 2, 2, 2, 1, 2, 2])
    assert np.allclose(res["t_unique_rel"], [2, 2, 2, 2, 2, 2, 2, 2])
    assert np.allclose(res["t_degree"], [3, 3, 3, 3, 2, 2, 3, 3])
    assert np.allclose(res["t_degree_same_rel"], [1, 2, 2, 1, 1, 1, 2, 2])
    assert np.allclose(res["tot_degree"], [4, 4, 5, 4, 3, 3, 4, 5])
    assert np.allclose(res["tot_degree_same_rel"], [2, 2, 3, 2, 2, 1, 3, 3])

    # triple cardinality
    assert res["triple_cardinality"].tolist() == [
        "M:M",
        "M:M",
        "M:M",
        "M:M",
        "M:M",
        "M:M",
        "M:M",
        "M:M",
    ]
    assert res["triple_cardinality_same_rel"].tolist() == [
        "1:M",
        "M:1",
        "M:M",
        "1:M",
        "1:M",
        "1:1",
        "M:M",
        "M:M",
    ]

    # relation pattern symmetry
    res = kgtt.edge_pattern_summary(return_metapath_list=return_metapath_list)
    assert np.allclose(
        res["is_loop"], [False, False, False, False, False, False, True, True]
    )
    assert np.allclose(
        res["is_symmetric"], [False, False, True, False, True, False, False, False]
    )
    # relation pattern inverse
    assert np.allclose(
        res["has_inverse"], [False, False, True, False, False, True, False, False]
    )
    assert np.allclose(res["n_inverse_relations"], [0, 0, 1, 0, 0, 1, 0, 0])
    # relation pattern inference
    assert np.allclose(
        res["has_inference"], [True, True, False, False, True, True, False, False]
    )
    assert np.allclose(res["n_inference_relations"], [1, 1, 0, 0, 1, 1, 0, 0])

    # relation_pattern_composition & metapaths
    assert np.allclose(
        res["has_composition"], [False, False, True, False, False, False, False, False]
    )
    assert np.allclose(res["n_triangles"], [0, 0, 2, 0, 0, 0, 0, 0])
    assert np.allclose(res["n_undirected_triangles"], [3, 3, 2, 6, 2, 2, 0, 0])
    if return_metapath_list:
        assert res["metapath_list"][2] == ["0-1", "1-1"]


def test_filter_relations() -> None:
    for rels in [[0], [1], [0, 1]]:
        for method in [
            kgtt.edge_degree_cardinality_summary,
            partial(kgtt.edge_pattern_summary, return_metapath_list=True),
        ]:
            # compare outputs of standard method call and filtered call
            res_all = method()
            res_all = res_all[res_all.r.isin(rels)]
            res_filtered = method(filter_relations=rels)
            assert np.all(res_all.index.values == res_filtered.index.values)
            for c in res_all.columns:
                if c == "metapath_list":
                    for a, b in zip(res_all[c].values, res_filtered[c].values):
                        assert a == b
                else:
                    assert np.all(res_all[c].values == res_filtered[c].values)
