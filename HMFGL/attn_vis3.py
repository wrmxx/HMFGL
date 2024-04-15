import networkx as nx
import matplotlib.pyplot as plt
import argparse
import  numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--datname', type=str, default='TADPOLE')

args = parser.parse_args()

data_name = args.datname
label = np.load('./graph/{}_weighted-cosine_graph_experiment6.npz'.format(data_name))['label']
fused = np.load('./graph/{}_weighted-cosine_graph_experiment6.npz'.format(data_name))['fused']
feat = np.load('./graph/{}_weighted-cosine_graph_experiment6.npz'.format(data_name))['embedding']
A = np.load('./graph/{}_weighted-cosine_graph_experiment6.npz'.format(data_name))['adj']
G = nx.Graph()

for i in range(len(feat)):
    G.add_node(i)
for i in range(len(A)):
    for j in range(len(A)):
        if A[i][j] > 0:
            G.add_edge(i, j)

pos = nx.spring_layout(G)

nx.draw(G, pos, node_color='b', node_size=50)
plt.show()
