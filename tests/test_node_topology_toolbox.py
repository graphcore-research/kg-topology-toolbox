# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import pandas as pd
import pytest

from kg_topology_toolbox import KGTopologyToolbox

df = pd.DataFrame(
    dict(
        H=[0, 0, 0, 1, 2, 2, 2],
        T=[1, 1, 2, 2, 0, 0, 2],
        R=[0, 1, 0, 1, 0, 1, 1],
        n=["a", "b", "c", "d", "e", "f", "g"],
    )
)

kgtt = KGTopologyToolbox(df, head_column="H", relation_column="R", tail_column="T")


@pytest.mark.parametrize("return_relation_list", [True, False])
def test_node_degree_summary(return_relation_list: bool) -> None:
    # entity degrees statistics
    res = kgtt.node_degree_summary(return_relation_list=return_relation_list)
    assert np.allclose(res["h_degree"], [3, 1, 3])
    assert np.allclose(res["t_degree"], [2, 2, 3])
    assert np.allclose(res["tot_degree"], [5, 3, 5])
    assert np.allclose(res["h_unique_rel"], [2, 1, 2])
    assert np.allclose(res["t_unique_rel"], [2, 2, 2])
    assert np.allclose(res["n_loops"], [0, 0, 1])
    if return_relation_list:
        assert [x.tolist() for x in res["h_rel_list"].to_list()] == [
            [0, 1],
            [1],
            [0, 1],
        ]
        assert [x.tolist() for x in res["t_rel_list"].to_list()] == [
            [0, 1],
            [0, 1],
            [0, 1],
        ]
