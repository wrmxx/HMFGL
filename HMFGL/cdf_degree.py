import pandas as pd
from collections import Counter

import numpy as np
import scipy.io as scio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datname', type=str, default='TADPOLE')

args = parser.parse_args()

data_name = args.datname
adj_matrix = np.load('./graph/{}_weighted-cosine_graph_experiment6.npz'.format(data_name))['score']

A = adj_matrix > 0.85

node_degrees = np.sum(A, axis=1)

sorted_degrees = np.sort(node_degrees)
yvals = np.arange(len(sorted_degrees)) / float(len(sorted_degrees) - 1)

plt.figure(figsize=(8, 6))
plt.plot(sorted_degrees, yvals)
plt.xlabel('Node Degree')
plt.ylabel('CDF')
plt.title('Degree Cumulative Distribution Function (CDF)')
plt.grid(True)
plt.show()
