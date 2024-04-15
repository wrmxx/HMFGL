# Hybrid Multimodal Fusion for Graph Learning in Disease Prediction(HMFGL)

This is a PyTorch version of HMFGL model as proposed in our paper.

## Introduction

Graph neural networks (GNNs) have gained significant attention in disease prediction where the node represents patients' latent features and the edge denotes the similarity relationship between two patients. The graph structure plays a crucial role in graph learning by determining information aggregation and propagation. However, recent methods typically construct graphs based on patients' latent embeddings, which may not accurately reflect real-world connections. We observe meta data (e.g., demographic attributes, genetic markers) provide abundant information to gauge patient similarities and could compensate for the weakness of graph construction from latent representation. Therefore, we propose to initially create graphs from both latent representation and meta data respectively, and then combine these graphs through a weighted summation. Considering the graphs could include irrelevant and noisy connections, we employ the degree-sensitive edge pruning and kNN sparsification to sparsify and prune these edges. We conduct intensive experiments on two diagnosis prediction datasets and experiment results demonstrate that our proposed method outperforms state-of-the-art methods.


![HMFGL](https://github.com/JobYoo/HMFGL/assets/153283474/f56b4837-b9bd-4f38-b79c-a5b058fd1277)




## Requirements

* PyTorch = 1.9.1
* python 3.6
* networkx
* scikit-learn
* scipy
* munkres

## Code running
### Step 1: Data

The data preprocessing process are provided in [./data/{dataset}].
The download link for the TADPOLE dataset is https://tadpole.grand-challenge.org/Data/.

If you want to use your own data, you have to provide :

* a csv.file which contains multi-modal features, and
* a multi-modal feature dict.

### Step 2: Data prprocessing

Running the code of data preprocessing in ./data/{dataset}/xxx.ipynb to preprocess the raw data to standard data as the input of HMFGL.

### Step 3: Training and test

Running

```
./HMFGL/HMFGL-{dataset}.sh
```
