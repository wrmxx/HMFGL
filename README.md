# Hybrid Multimodal Fusion for Graph Learning in Disease Prediction(HMFGL)

This is a PyTorch version of HMFGL model as proposed in our paper.

## Introduction

Graph neural networks (GNNs) have gained significant attention in disease prediction where the latent embeddings of patients are modeled as nodes and the similarities among patients are represented through edges. The graph structure, which determines how information is aggregated and propagated, plays a crucial role in graph learning. 
Recent approaches typically create graphs based on patients' latent embeddings, which may not accurately reflect their real-world closeness. Our analysis reveals that raw data, such as demographic attributes and laboratory results, offers a wealth of information for assessing patient similarities and can serve as a compensatory measure for graphs constructed exclusively from latent embeddings. In this study, we first construct adaptive graphs from both latent representations and raw data respectively, and then merge these graphs via weighted summation. Given that the graphs may contain extraneous and noisy connections, we apply degree-sensitive edge pruning and kNN sparsification techniques to selectively sparsify and prune these edges. We conducted intensive experiments on two diagnostic prediction datasets, and the results demonstrate that our proposed method surpasses current state-of-the-art techniques.


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
