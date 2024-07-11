import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from biotite.sequence import NucleotideSequence

embeddings = []
with open(
    "../../Model Inference/Embeddings/hyenadna-large-1m-seqlen-hf.txt", "r+"
) as f:
    embeddingsDict = eval(f.read())
    for i in range(len(embeddingsDict)):
        embeddings.append(np.array(embeddingsDict[i]["embedding"]))

embeddings = np.array(embeddings)

clades = []
with open("../Data Labelling/labels.txt", "r+") as f:
    seqDict = eval(f.read())
    for seq in seqDict:
        clades.append(seq["clade"])


def T_SNE():
    # Adapted from https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/clustering_with_embeddings.ipynb#scrollTo=t-1QKCK8DHsI
    tsne = TSNE(random_state=0, max_iter=1000)
    tsneResults = tsne.fit_transform(embeddings)
    return tsneResults


def U_MAP():
    # Adapted from https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    reducer = umap.UMAP(
        random_state=42, n_epochs=1000
    )  # ensures reproducibility of umap results
    umapResults = reducer.fit_transform(embeddings)
    return umapResults


def plotter(method):
    if method == "UMAP":
        results = U_MAP()
        columns = ["UMAP1", "UMAP2"]
    elif method == "TSNE":
        results = T_SNE()
        columns = ["TSNE1", "TSNE2"]

    df = pd.DataFrame(results, columns=columns)
    df["Clades"] = clades

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.scatterplot(data=df, x=columns[0], y=columns[1], hue="Clades", palette="hls")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(f"Scatter plot of sequence embeddings using {method}")
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    plotter("UMAP")
    # plotter("TSNE")

# TODO: make a summary of all the new stuff you've learnt - dimensionality reduction, different libraries anol.
