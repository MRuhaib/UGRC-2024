import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import concurrent.futures
import time


hyenaModels = [
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    "LongSafari/hyenadna-small-32k-seqlen-hf",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]

models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    "zhihan1996/DNABERT-S",
    "zhihan1996/DNABERT-2-117M",
]

newModels = [
    "songlab/gpn-brassicales",
    "PoetschLab/GROVER",
]

genes = [
    "YAL001C",
    "YAL007C",
    "YAL018C",
    "YAL022C",
    "YAL026C",
    "YJR136C",
    "YPR192W",
]

# iterate over all genes -> pairwise euclidean calculation for each model's embeddings for that gene -> store accordingly in the euclid embedding alignment scores folder.


def main(model, gene):
    start = time.time()
    modelEmbeddings = []
    with open(
        f"C:/Users/Ruhaib/Downloads/IIT Stuff/Research/Nirav Sir's Lab/Model Evaluation/Model Inference/Embeddings/{gene}/{(model).split('/')[1]}_{gene}.txt",  # remove Hyena
        "r+",
    ) as f:  # add hyena in a separate run.
        modelEmbeddings = eval(f.read())
    print(
        f"Done with pairwise comparison of {(model).split('/')[1]}'s embeddings for gene {gene}"
    )
    embeddings = []
    for element in modelEmbeddings:
        embeddings.append(element["embedding"])
    alignment = squareform(pdist(embeddings, "euclidean"))
    embeddingAlignmentScores = pd.DataFrame(alignment)
    embeddingAlignmentScores.to_csv(
        f"Embedding Alignment Scores/Euclidean/{gene}/{(model).split('/')[1]}_{gene}_distances.csv",  # remove hyena
        index=False,
    )  # add hyena
    finish = time.time()
    print(
        f"Done with pairwise comparison of {(model).split('/')[1]}'s {gene} embeddings in {round(finish - start)} seconds."
    )


if __name__ == "__main__":

    for gene in genes:
        for model in newModels:
            main(model, gene)

    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        results = executor.map(main, models)
        for result in results:
            print(result)
    """
