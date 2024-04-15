import numpy as np
import scipy.io as scio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import normalize
import seaborn as sns
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument('--datname', type=str, default='ABIDE')

args = parser.parse_args()

data_name = args.datname
HMFGL_TADPOLE_label = np.load('./graph/TADPOLE_weighted-cosine_graph_experiment6.npz')['label']
HMFGL_TADPOLE_fused = np.load('./graph/TADPOLE_weighted-cosine_graph_experiment6.npz')['fused']
HMFGL_TADPOLE_em = np.load('./graph/TADPOLE_weighted-cosine_graph_experiment6.npz')['embedding']
HMFGL_TADPOLE_adjs = np.load('./graph/TADPOLE_weighted-cosine_graph_experiment6.npz')['adj']

HMFGL_ABIDE_label = np.load('./graph/ABIDE_weighted-cosine_graph_experiment6.npz')['label']
HMFGL_ABIDE_fused = np.load('./graph/ABIDE_weighted-cosine_graph_experiment6.npz')['fused']
HMFGL_ABIDE_em = np.load('./graph/ABIDE_weighted-cosine_graph_experiment6.npz')['embedding']
HMFGL_ABIDE_adjs = np.load('./graph/ABIDE_weighted-cosine_graph_experiment6.npz')['adj']

MMGL_TADPOLE_label = np.load('./graph/TADPOLE_weighted-cosine_graph_.npz')['label']
MMGL_TADPOLE_fused = np.load('./graph/TADPOLE_weighted-cosine_graph_.npz')['fused']
MMGL_TADPOLE_em = np.load('./graph/TADPOLE_weighted-cosine_graph_.npz')['embedding']
MMGL_TADPOLE_adjs = np.load('./graph/TADPOLE_weighted-cosine_graph_.npz')['adj']

MMGL_ABIDE_label = np.load('./graph/ABIDE_weighted-cosine_graph_.npz')['label']
MMGL_ABIDE_fused = np.load('./graph/ABIDE_weighted-cosine_graph_.npz')['fused']
MMGL_ABIDE_em = np.load('./graph/ABIDE_weighted-cosine_graph_.npz')['embedding']
MMGL_ABIDE_adjs = np.load('./graph/ABIDE_weighted-cosine_graph_.npz')['adj']



HMFGL_TADPOLE_unique_labels = np.unique(HMFGL_TADPOLE_label)
HMFGL_ABIDE_unique_labels = np.unique(HMFGL_ABIDE_label)
MMGL_TADPOLE_unique_labels = np.unique(MMGL_TADPOLE_label)
MMGL_ABIDE_unique_labels = np.unique(MMGL_ABIDE_label)


HMFGL_TADPOLE_fused_reordered_feat = []
HMFGL_TADPOLE_em_reordered_feat = []

HMFGL_ABIDE_fused_reordered_feat = []
HMFGL_ABIDE_em_reordered_feat = []

MMGL_TADPOLE_fused_reordered_feat = []
MMGL_TADPOLE_em_reordered_feat = []

MMGL_ABIDE_fused_reordered_feat = []
MMGL_ABIDE_em_reordered_feat = []


#HMFGL TADPOLE fused
for label_value in HMFGL_TADPOLE_unique_labels:
    indices = np.where(HMFGL_TADPOLE_label == label_value)[0]
    HMFGL_TADPOLE_fused_reordered_feat.extend(HMFGL_TADPOLE_fused[indices])
#HMFGL TADPOLE em
for label_value in HMFGL_TADPOLE_unique_labels:
    indices = np.where(HMFGL_TADPOLE_label == label_value)[0]
    HMFGL_TADPOLE_em_reordered_feat.extend(HMFGL_TADPOLE_em[indices])

#HMFGL ABIDE fused
for label_value in HMFGL_ABIDE_unique_labels:
    indices = np.where(HMFGL_ABIDE_label == label_value)[0]
    HMFGL_ABIDE_fused_reordered_feat.extend(HMFGL_ABIDE_fused[indices])

#HMFGL ABIDE em
for label_value in HMFGL_ABIDE_unique_labels:
    indices = np.where(HMFGL_ABIDE_label == label_value)[0]
    HMFGL_ABIDE_em_reordered_feat.extend(HMFGL_ABIDE_em[indices])

#MMGL TADPOLE fused
for label_value in MMGL_TADPOLE_unique_labels:
    indices = np.where(MMGL_TADPOLE_label == label_value)[0]
    MMGL_TADPOLE_fused_reordered_feat.extend(MMGL_TADPOLE_fused[indices])
#MMGL TADPOLE em
for label_value in MMGL_TADPOLE_unique_labels:
    indices = np.where(MMGL_TADPOLE_label == label_value)[0]
    MMGL_TADPOLE_em_reordered_feat.extend(MMGL_TADPOLE_em[indices])

#MMGL ABIDE fused
for label_value in MMGL_ABIDE_unique_labels:
    indices = np.where(MMGL_ABIDE_label == label_value)[0]
    MMGL_ABIDE_fused_reordered_feat.extend(MMGL_ABIDE_fused[indices])

for label_value in MMGL_ABIDE_unique_labels:
    indices = np.where(MMGL_ABIDE_label == label_value)[0]
    MMGL_ABIDE_em_reordered_feat.extend(MMGL_ABIDE_em[indices])

HMFGL_TADPOLE_fused_cosine_sim = cosine_similarity(HMFGL_TADPOLE_fused_reordered_feat)
HMFGL_TADPOLE_em_cosine_sim = cosine_similarity(HMFGL_TADPOLE_em_reordered_feat)
HMFGL_ABIDE_fused_cosine_sim = cosine_similarity(HMFGL_ABIDE_fused_reordered_feat)
HMFGL_ABIDE_em_cosine_sim = cosine_similarity(HMFGL_ABIDE_em_reordered_feat)

MMGL_TADPOLE_fused_cosine_sim = cosine_similarity(MMGL_TADPOLE_fused_reordered_feat)
MMGL_TADPOLE_em_cosine_sim = cosine_similarity(MMGL_TADPOLE_em_reordered_feat)
MMGL_ABIDE_fused_cosine_sim = cosine_similarity(MMGL_ABIDE_fused_reordered_feat)
MMGL_ABIDE_em_cosine_sim = cosine_similarity(MMGL_ABIDE_em_reordered_feat)

HMFGL_cosine_sims = [
    HMFGL_TADPOLE_fused_cosine_sim,
    HMFGL_TADPOLE_em_cosine_sim,
    HMFGL_ABIDE_fused_cosine_sim,
    HMFGL_ABIDE_em_cosine_sim
]

MMGL_cosine_sims = [
    MMGL_TADPOLE_fused_cosine_sim,
    MMGL_TADPOLE_em_cosine_sim,
    MMGL_ABIDE_fused_cosine_sim,
    MMGL_ABIDE_em_cosine_sim
]

title1 = ['(e)','(f)','(g)','(h)']
title2 = ['(a)','(b)','(c)','(d)']

fig = plt.figure(figsize=(20, 5), dpi=300)
gs = gridspec.GridSpec(1, 5, width_ratios=[ 1, 1, 1, 1, 0.05])

for i, cosine_sim in enumerate(HMFGL_cosine_sims):
    ax = plt.subplot(gs[0, i])
    im = ax.imshow(cosine_sim, cmap='plasma', vmin=0, vmax=1)
    ax.axis('on')
    x_interval = 200
    y_interval = 200
    x_labels = list(range(0, len(cosine_sim), x_interval))
    y_labels = list(range(0, len(cosine_sim), y_interval))

    ax.set_xticks(x_labels)
    ax.set_yticks(y_labels)
    ax.set_xticklabels(x_labels, fontsize=20)
    ax.set_yticklabels(y_labels, fontsize=20)
    ax.text(0.5, -0.1, title1[i], ha='center', va='top', fontsize=20, transform=ax.transAxes)


cax = plt.subplot(gs[0, -1],label='colorbar_ax')
cbar = plt.colorbar(im, cax=cax,pad=0.005)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()

plt.show()
