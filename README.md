# KG Topology Toolbox
![Continuous integration](https://github.com/graphcore-research/kg-topology-toolbox/actions/workflows/ci.yaml/badge.svg)

A Python toolbox to compute topological metrics and statistics for Knowledge Graphs.

Documentation can be found at https://graphcore-research.github.io/kg-topology-toolbox/

For a walkthrough of the main functionalities, we provide an introductory [Jupyter notebook](docs/source/notebooks/ogb_biokg_demo.ipynb).

## Usage

Tested on Ubuntu 20.04, Python >=3.9

To install the `kg-topology-toolbox` library, run

```
pip install wheel
pip install git+https://github.com/graphcore-research/kg-topology-toolbox.git
```

4\. Import and use:
```python
from kg_topology_toolbox import KGTopologyToolbox
```

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under the MIT license (see [details of the license](LICENSE)).

See [notices](NOTICE.md) for dependencies, credits, derived work and further details.
