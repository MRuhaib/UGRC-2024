import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    OPTICS,
    AffinityPropagation,
)
import concurrent.futures
from biotite.sequence import NucleotideSequence

"""
models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]

models = [
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    "LongSafari/hyenadna-small-32k-seqlen-hf",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]
"""
models = [
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]


def T_SNE(embeddings):
    # Adapted from https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/clustering_with_embeddings.ipynb#scrollTo=t-1QKCK8DHsI
    params = {"random_state": 0, "max_iter": 1000}
    tsne = TSNE(random_state=params["random_state"], max_iter=params["max_iter"])
    tsneResults = tsne.fit_transform(embeddings)
    return tsneResults, params


def U_MAP(embeddings):
    params = {
        "random_state": 42,
        "n_neighbors": 30,
        "n_epochs": None,
        "learning_rate": 0.01,
        "min_dist": 0.5,
        "spread": 7,
    }
    # Adapted from https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    reducer = umap.UMAP(
        random_state=params["random_state"],  # ensures reproducibility of umap results
        n_neighbors=params["n_neighbors"],
        # n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        min_dist=params["min_dist"],
        spread=params["spread"],
        metric="manhattan",
    )
    umapResults = reducer.fit_transform(embeddings)

    return umapResults, params


def plotter(model, df, method, cluster, algo=None):
    if method == "UMAP":
        results, params = U_MAP([[0]])
        columns = ["UMAP1", "UMAP2"]
    elif method == "TSNE":
        results, params = T_SNE([[0]])
        columns = ["TSNE1", "TSNE2"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.scatterplot(data=df, x=columns[0], y=columns[1], hue=cluster, palette="hls")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(
        f"Scatter plot of {model}'s embeddings using {f'{algo} clustering' if cluster == 'Cluster' else method}"
    )
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.axis("equal")

    plt.show()
    save = "y"  # input(f"Save {model}'s plot? y/n")
    if save == "y":
        fig.savefig(
            f"Results/{method}/{'Clustering' if cluster == 'Cluster' else 'param testing'}/{algo if cluster == 'Cluster' else method} {' '.join([f'{key}-{value}' for key, value in params.items()])}.png"
        )


def cluster(model, embeddings, df, method, algo):
    # read: https://scikit-learn.org/stable/modules/clustering.html
    if algo == "KMeans":
        clusteringModel = KMeans(n_clusters=30, random_state=1).fit(embeddings)
    elif algo == "Agglomerative":
        clusteringModel = AgglomerativeClustering(n_clusters=30).fit(embeddings)
    elif algo == "Birch":
        clusteringModel = Birch(n_clusters=30).fit(embeddings)
    elif algo == "DBSCAN":
        clusteringModel = DBSCAN().fit(embeddings)
    elif algo == "OPTICS":
        clusteringModel = OPTICS().fit(embeddings)
    elif algo == "AffinityPropagation":
        clusteringModel = AffinityPropagation().fit(embeddings)
    labels = clusteringModel.fit_predict(embeddings)
    df["Cluster"] = labels
    plotter(model, df, method, "Cluster", algo)


def main(model, method="UMAP"):
    embeddings = []
    model = model.split("/")[1]
    with open(
        f"../../Model Inference/Embeddings/{model}.txt", "r+"
    ) as f:  # remove hyena
        embeddingsDict = eval(f.read())
        for i in range(len(embeddingsDict)):
            embeddings.append(np.array(embeddingsDict[i]["embedding"]))

    embeddings = np.array(embeddings)

    clades = []
    with open("../Data Labelling/labels.txt", "r+") as f:
        seqDict = eval(f.read())
        for seq in seqDict:
            clades.append(seq["clade"])

    # method = input("Enter dimension reduction method:").upper()

    if method == "UMAP":
        results, params = U_MAP(embeddings)
        columns = ["UMAP1", "UMAP2"]
    elif method == "TSNE":
        results, params = T_SNE(embeddings)
        columns = ["TSNE1", "TSNE2"]

    df = pd.DataFrame(results, columns=columns)
    df["Clades"] = clades

    plotter(model, df, method, "Clades")
    """
    algos = [
        "KMeans",
        "Agglomerative",
        "OPTICS",
        "Birch",  # useless
        "DBSCAN",  # useless
        "AffinityPropagation",
    ]
    for algo in algos:
        cluster(model, embeddings, df, method, algo)
    """
    return f"Done with {model}"


# Make functions for 2-3 different clustering techniques and use them


if __name__ == "__main__":
    print(main(models[0]))

    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(main, models)
        for result in results:
            print(result)
    """

# TODO: make a summary of all the new stuff you've learnt - dimensionality reduction, different libraries anol.
