import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import concurrent.futures
import time

"""
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
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]


def main(model):
    start = time.time()
    modelEmbeddings = []
    with open(
        f"../../../Embeddings/{(model).split('/')[1]}.txt", "r+"
    ) as f:  # add hyena
        modelEmbeddings = eval(f.read())
    embeddings = []
    for element in modelEmbeddings:
        embeddings.append(element["embedding"])
    alignment = squareform(pdist(embeddings, "euclidean"))
    embeddingAlignmentScores = pd.DataFrame(alignment)
    embeddingAlignmentScores.to_csv(f"{(model).split('/')[1]}Scores.csv")  # add hyena
    finish = time.time()
    return f"Done with pairwise comparison of {(model).split('/')[1]}'s embeddings in {round(finish - start)} seconds."


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        results = executor.map(main, models)
        for result in results:
            print(result)
