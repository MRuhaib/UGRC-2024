import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import pdist, squareform
import concurrent.futures
from biotite.sequence import NucleotideSequence


def HAC(embeddings, labels):
    distances = pdist(embeddings, metric="euclidean")
    clusters = linkage(distances, method="single", optimal_ordering=True)
    clusterNode = to_tree(clusters)
    fig = plt.figure(figsize=(10, 10))
    dn = dendrogram(clusters)
    results = dn["ivl"]
    labels = [labels[int(seq) - 1] for seq in results]
    dn = dendrogram(
        clusters,
        p=30,
        # truncate_mode="level",
        labels=labels,
        orientation="right",
    )
    plt.show()


def main(model):
    embeddings = []
    model = model.split("/")[1]
    with open(f"../../Model Inference/Embeddings/{model}.txt", "r+") as f:
        embeddingsDict = eval(f.read())
        for i in range(len(embeddingsDict)):
            embeddings.append(np.array(embeddingsDict[i]["embedding"]))

    embeddings = np.array(embeddings)

    clades = []
    with open("../Data Labelling/labels.txt", "r+") as f:
        seqDict = eval(f.read())
        for seq in seqDict:
            clades.append(seq["clade"])

    HAC(embeddings, clades)
    print(f"Done with {model}")


if __name__ == "__main__":
    main("LongSafari/hyenadna-large-1m-seqlen-hf")
