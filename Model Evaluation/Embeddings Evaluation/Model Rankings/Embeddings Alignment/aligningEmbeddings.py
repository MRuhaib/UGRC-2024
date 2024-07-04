import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import time


models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]

modelEmbeddings = []
modelCount = 0  # needed to make this a global variable so it could be sued for indexing in rowAdder, but I suppose there might be a more easier, logical way to do it.

for model in models:
    name = model.split("/")[1]
    with open(f"../Embeddings/{name}.txt", "r+") as f:
        modelDict = eval(f.read())
        modelEmbeddings.append(modelDict)


def rowAdder(i):
    # i must correspond to one specific sequence and its embedding!

    row = np.zeros(len(modelEmbeddings[modelCount]))
    for j in range(i, len(modelEmbeddings[modelCount])):
        embedding1, embedding2 = (
            modelEmbeddings[modelCount][i]["embedding"],
            modelEmbeddings[modelCount][j]["embedding"],
        )
        row[j] = cosine_similarity(embedding1, embedding2)
    if i % 100 == 0:
        print(
            f"Done with {i} rows for {(models[modelCount]).split('/')[1]}'s embeddings comparison"
        )
    return i, row


if __name__ == "__main__":

    for sequence in modelEmbeddings:
        start = time.time()
        embeddingAlignmentScores = pd.DataFrame(
            columns=[i for i in range(len(sequence))]
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
            results = executor.map(rowAdder, range(len(modelEmbeddings[modelCount])))
            for result in results:
                i, row = result
                embeddingAlignmentScores[i] = row

        for i in range(len(modelEmbeddings[modelCount])):
            for j in range(len(modelEmbeddings[modelCount])):
                alignmentScores.iloc[i, j] = alignmentScores.iloc[j, i]

        embeddingAlignmentScores.to_csv(
            f"Sequence Alignment Scores/{(models[modelCount]).split('/')[1]}.csv"
        )
        finish = time.time()
        print(
            f"Done with pairwise comparison of {(models[modelCount]).split('/')[1]}'s embeddings in {round(finish - start)} seconds."
        )
        modelCount += 1
