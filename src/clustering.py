import argparse
import os
import json
import sys

import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

from tqdm import tqdm

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--feature", required=True)
parser.add_argument("--save", default="./clustering")

args = parser.parse_args()

# prop = args.property
save_path = args.save
feature_dir = args.feature # json
feature_file = f"{feature_dir}/features.json"


if __name__ == "__main__":
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.read_json(feature_file)
    print(f"{df=}")
    df.to_csv(os.path.join(save_path, "data_frame.csv"))
    names = [id_.replace("POSCAR_", "").replace(".vasp", "") for id_ in tqdm(df.id)]
    layers = ["ALIGNN1", "ALIGNN2", "ALIGNN3", "ALIGNN4", "CGN1", "CGN2", "CGN3", "CGN4",
              "Last1", "prediction", "reference"]
    n_cluster = 100
    for layer in layers:
        print(f"clustering {layer=} ...")
        z = df[layer].to_list()
        z = np.array(z)
        distances = pdist(z, metric='euclidean')
        linkage_matrix = linkage(distances, method='ward')
        save_path_l = os.path.join(save_path, layer)
        os.mkdir(os.path.join(save_path_l))
        np.savez(f'{save_path_l}/clustering_data.npz',
                 linkage_matrix=linkage_matrix,
                 sample_names=names,
                 ids=df.index.tolist())
        threshold_distance = linkage_matrix[-(n_cluster - 1), 2]

        # Dendrogram plot
        plt.figure(figsize=(80, 42))
        # dendrogram(Z, color_threshold=threshold_distance, labels=df.index.to_list(), )
        dendrogram(linkage_matrix, color_threshold=threshold_distance, labels=names)
        plt.savefig(f"{save_path_l}/dendrogram.pdf")

