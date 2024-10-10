import numpy as np
import pandas as pd
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

def generateRanking(index, scores):
    row = scores.loc[index]
    seqScores = []
    ranking = []
    for i, score in enumerate(row):
        if i != index:
            seqScores.append(score)
        else:
            seqScores.append(
                10000
            )  # So that the current 'anchor' sequence is not considered in the ranking; it gets popped out of the list later.
    seqScores = np.argsort(
        seqScores
    )  # Since the sequence number is the same as the score's index in the row, this returns a sorted array with the indices corresponding to the sorted scores.
    for i in seqScores:
        ranking.append(i)
    ranking.pop()
    return {"seq": index, "ranking": ranking}

def main(modelCount):
    length = 1011 #straight up hard coding it, reading the file each time to find the length takes time.
    start = time.time()
    embeddingAlignmentScores = pd.read_csv(f"../../Single Gene Embeddings/Embeddings Alignment/Embedding Alignment Scores/Filled/{(models[modelCount]).split('/')[1]}Scores.csv")

    rankings = []

    for index in range(length):
        rankings.append(generateRanking(index, embeddingAlignmentScores))

    with open(f"{(models[modelCount]).split('/')[1]}Ranking.txt", "w+") as f:
        f.write(str(rankings))
    finish = time.time()
    return f'Finished ranking the embedding alignment scores for {(models[modelCount]).split('/')[1]} in {round(finish - start)} seconds.'

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        results = executor.map(main, range(len(models)))
        for result in results:
            print(result)

