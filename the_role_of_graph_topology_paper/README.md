# The Role of Graph Topology in the Performance of Biomedical Knowledge Graph Completion Models

Code to reproduce results of [The Role of Graph Topology in the Performance of Biomedical Knowledge Graph Completion Models](https://arxiv.org/abs/2409.04103).

**Usage:**

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements_paper.txt
```

 For each directory in `datasets/` (one per Knowledge Graph), run the preprocessing notebook to download and preprocess the KG dataset. The file `test_ranks.csv` contains the ranks of tail predictions on the test set for all KGE models trained in the paper.
 
To reproduce the experiments from the paper, see [train/train.py](train/train.py) and the scripts provided in [`train/scripts`](train/scripts/). This is dependant on the [BESS-KGE](https://github.com/graphcore-research/bess-kge) package: please refer to the repo's README for setting it up.
```
cd train/
bash scripts/train_hetionet.sh
```

We also provide notebooks for all the data analysis and visualisations:
 
- [compute_topology_metrics](notebooks/compute_topology_metrics.ipynb): compute the topological metrics of all Knowledge Graphs, using the [kg-topology-toolbox](https://github.com/graphcore-research/kg-topology-toolbox) package.
- [hetionet_pharmebinet_link](notebooks/hetionet_pharmebinet_link.ipynb): link common triples and relation types in Hetionet and PharMeBINet.
- [visualisations](notebooks/visualisations.ipynb): reproduce the data analysis and visualisations from the paper.
