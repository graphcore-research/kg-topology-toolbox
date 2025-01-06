<h1 align="center">
  KG Topology Toolbox
</h1>

![Continuous integration](https://github.com/graphcore-research/kg-topology-toolbox/actions/workflows/ci.yaml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<b>KG Topology Toolbox:</b> is a Python package designed to efficiently extract common topological patterns from knowledge graphs, 
including edge patterns (e.g. symmetric, inverse, inference & composition) and edge cardinalities (e.g. one-to-one, one-to-many, many-to-one, many-to-many).

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)


`kg-topology-toolbox` is a Python-based toolbox for computing topological properties of Knowledge Graphs (KGs). This library provides researchers and practitioners with tools to better understand the structural characteristics of KGs and how they might impact the predictive performance of any models trained upon them.

`kg-topology-toolbox` enables computation of a variety of properties of KGs, focusing on **edge topological (symmetry, inverse, inference, loop and composition) and cardinality patterns (one-to-one, one-to-many, many-to-one & many-to-many).** 

Unlike other libraries, `kg-topology-toolbox` provides the ability to compute these properties at the level of individual triples, as well as at the level of relations.

![edge patterns](docs/source/images/edge_patterns.png "Edge Patterns")

---

Full documentation can be found at https://graphcore-research.github.io/kg-topology-toolbox/

> For a walkthrough of the main functionalities of `kg-topology-toolbox`, we provide an introductory [Jupyter notebook](docs/source/notebooks/ogb_biokg_demo.ipynb). 

## Installation

The library has been tested on Ubuntu 20.04 & MacOS >= 14 and has been developed targetting Python >=3.9 - however it should be widely compatible with other systems.

To install the latest version of `kg-topology-toolbox` library, run:

```
pip install wheel
pip install git+https://github.com/graphcore-research/kg-topology-toolbox.git
```

If you would like to be able to change the source code and have the changes reflected in your environment, you can clone the repository and install the package in editable mode run:

```
git clone https://github.com/graphcore-research/kg-topology-toolbox.git
cd kg-topology-toolbox
pip install -e .
```

## Usage

Once installed, the library can be imported as follows:

```python
from kg_topology_toolbox import KGTopologyToolbox
```

`kg-topology-toolbox` requires that the input KG is in the form of a pandas DataFrame with suggested column names of `h`, `r` and `t`. The `h` and `t` columns should contain the head and tail entities involved in the triple, and the `r` column should contain the relation type. These columns should be the integer identifiers of the entities and relations in the KG. Note that if your columns are named differently, you can specify the column names when creating the `KGTopologyToolbox` object.

For example, we can load a KG from a CSV file:

```python
import pandas as pd
df = pd.read_csv("path/to/kg.csv", columns=["h", "r", "t"])
```

This can then be used to instantiate a `KGTopologyToolbox` object:

```python
kgtt = KGTopologyToolbox(df)
```

### Computing Edge Topological Patterns

The `KGTopologyToolbox` object can be used to compute the topological properties of the KG. For example, to compute the edge patterns of the KG, we can use the [`edge_pattern_summary`](https://graphcore-research.github.io/kg-topology-toolbox/generated/kg_topology_toolbox.topology_toolbox.KGTopologyToolbox.html#kg_topology_toolbox.topology_toolbox.KGTopologyToolbox.edge_pattern_summary) method:

```python
edge_eps = kgtt.edge_pattern_summary()
```

This will return a DataFrame with the edge patterns of the KG, where values have been computed for each edge contained within the graph.

The values computed by the [`edge_pattern_summary`](https://graphcore-research.github.io/kg-topology-toolbox/generated/kg_topology_toolbox.topology_toolbox.KGTopologyToolbox.html#kg_topology_toolbox.topology_toolbox.KGTopologyToolbox.edge_pattern_summary) method include: loop, symmetric, inverse, inference, composition, number of triangles and other pattern metrics.

### Computing Edge Cardinality Patterns

Similarly, to compute the cardinality patterns of the KG, we can use the [`edge_degree_cardinality_summary`](https://graphcore-research.github.io/kg-topology-toolbox/generated/kg_topology_toolbox.topology_toolbox.KGTopologyToolbox.html#kg_topology_toolbox.topology_toolbox.KGTopologyToolbox.edge_degree_cardinality_summary) method:

```python
edge_dcs = kgtt.edge_degree_cardinality_summary()
```

This will return a DataFrame with the cardinality patterns of the KG, where again values have been computed for each edge contained within the graph. 

The values computed by the [`edge_degree_cardinality_summary`](https://graphcore-research.github.io/kg-topology-toolbox/generated/kg_topology_toolbox.topology_toolbox.KGTopologyToolbox.html#kg_topology_toolbox.topology_toolbox.KGTopologyToolbox.edge_degree_cardinality_summary) method include triple cardinality (one-to-one, one-to-many, many-to-one, many-to-many), head and tail degrees and other cardinality metrics.

### Aggregating by Relation

It is also possible to aggregate the properties at the level of relations, you can use the [`aggregate_by_relation`](https://graphcore-research.github.io/kg-topology-toolbox/generated/kg_topology_toolbox.utils.aggregate_by_relation.html#kg_topology_toolbox.utils.aggregate_by_relation) method:

```python
from kg_topology_toolbox.utils import aggregate_by_relation

relation_eps = aggregate_by_relation(edge_eps)
relation_dcs = aggregate_by_relation(edge_dcs)
```

This will return a DataFrame with statistics for the edge properties, aggregated across edges of the same relation type, for all relations contained within the graph.


**For a more detailed overview of the functionalities of `kg-topology-toolbox`, please refer to the [documentation](https://graphcore-research.github.io/kg-topology-toolbox/) and the introductory [Jupyter notebook](docs/source/notebooks/ogb_biokg_demo.ipynb).**

## Citation

If you have found this package useful in your research, please consider citing
[our paper](https://arxiv.org/abs/2409.04103):

```bibtex
@article{cattaneo2024role,
  title={The Role of Graph Topology in the Performance of Biomedical Knowledge Graph Completion Models},
  author={Cattaneo, Alberto and Bonner, Stephen and Martynec, Thomas and Luschi, Carlo and Barrett, Ian P and Justus, Daniel},
  journal={arXiv preprint arXiv:2409.04103},
  year={2024}
}
```

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under the MIT license (see [details of the license](LICENSE)).

See [notices](NOTICE.md) for dependencies, credits, derived work and further details.
