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

def main(modelCount):
    length = 1011 #straight up hard coding it, reading the file each time to find the length takes time.
    start = time.time()
    embeddingAlignmentScores = pd.read_csv(f"Embedding Alignment Scores/Raw/{(models[modelCount]).split('/')[1]}Scores.csv")
    for i in range(length):
        for j in range(length):
            embeddingAlignmentScores.iloc[i, j] = embeddingAlignmentScores.iloc[j, i]
    embeddingAlignmentScores.to_csv(
        f"Embedding Alignment Scores/Filled/{(models[modelCount]).split('/')[1]}Scores.csv"
    )
    finish = time.time()
    return f'Finished filling the embedding alignment scores for {(models[modelCount]).split('/')[1]} in {round(finish - start)} seconds.'

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        results = executor.map(main, range(len(models)))
        for result in results:
            print(result)
