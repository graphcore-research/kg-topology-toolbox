# KG Topology Toolbox
![Continuous integration](https://github.com/graphcore-research/kg-topology-toolbox/actions/workflows/ci.yaml/badge.svg)

A Python toolbox to compute topological metrics and statistics for Knowledge Graphs.

Documentation can be found at https://curly-barnacle-lnejye6.pages.github.io/

For a walkthrough of the main functionalities, we provide an introductory [Jupyter notebook](docs/source/notebooks/ogb_biokg_demo.ipynb).

## Usage

Tested on Ubuntu 20.04, Python >=3.8

To install the `kg-topology-toolbox` library, run

```
pip install wheel
pip install git+ssh://git@github.com/graphcore-research/kg-topology-toolbox
```

4\. Import and use:
```python
from kg_topology_toolbox import KGTopologyToolbox
```