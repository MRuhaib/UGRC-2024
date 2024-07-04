import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import concurrent.futures
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]

for model in models:
    name = model.split("/")[1]
    with open(f"../Embeddings/{name}.txt", "r+") as f:
        modelDict = eval(f.read())
        modelEmbeddings.append(modelDict)


def cos_sim(embedding1, embedding2):
    if torch.cuda.is_available():
        embedding1, embedding2 = (
            torch.tensor(embedding1).to(device),
            torch.tensor(embedding2).to(device),
        )
        e1normalized, e2normalized = torch.nn.functional.normalize(
            embedding1
        ), torch.nn.functional.normalize(embedding2)
        return torch.mm(e1normalized, e2normalized.transpose(0, 1)).numpy()
        # Try the gpu as a last case resort.
    else:
        return cosine_similarity([embedding1], [embedding2])


# Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py, so as to efficiently use the gpu.


def rowAdder(i, modelEmbeddings, modelCount):
    # i must correspond to one specific sequence and its embedding!

    row = np.zeros(len(modelEmbeddings))
    for j in range(i, len(modelEmbeddings)):
        embedding1, embedding2 = (
            modelEmbeddings[i]["embedding"],
            modelEmbeddings[j]["embedding"],
        )
        row[j] = cos_sim(embedding1, embedding2).squeeze()
    if i % 50 == 0:
        print(
            f"Done with {i} rows for {(models[modelCount]).split('/')[1]}'s embeddings comparison"
        )
    return row


def main(modelCount):
    start = time.time()
    modelEmbeddings = []
    with open(f"../Embeddings/{(models[modelCount]).split('/')[1]}.txt", "r+") as f:
        modelEmbeddings = eval(f.read())
    print(f"Starting embeddings alignment for {(models[modelCount]).split('/')[1]}.")
    embeddingAlignmentScores = pd.DataFrame(
        columns=[i for i in range(len(modelEmbeddings))]
    )
    for i in range(len(modelEmbeddings)):
        row = rowAdder(i, modelEmbeddings, modelCount)
        embeddingAlignmentScores[i] = row
    embeddingAlignmentScores.to_csv(
        f"Embedding Alignment Scores/Raw/{(models[modelCount]).split('/')[1]}Scores.csv"
    )
    finish = time.time()
    return f"Done with pairwise comparison of {(models[modelCount]).split('/')[1]}'s embeddings in {round(finish - start)} seconds."


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        results = executor.map(main, range(len(models)))
        for result in results:
            print(result)
