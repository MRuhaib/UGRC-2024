import pandas as pd
import concurrent.futures
import time
'''
models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]
'''
models = [
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    "LongSafari/hyenadna-small-32k-seqlen-hf",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]

def main(model):
    length = 1011 #straight up hard coding it, reading the file each time to find the length takes time.
    start = time.time()
    embeddingAlignmentScores = pd.read_csv(f"Embedding Alignment Scores/Raw/Hyena/{(model).split('/')[1]}Scores.csv") # remove hyena
    for i in range(length):
        for j in range(length):
            embeddingAlignmentScores.iloc[i, j] = embeddingAlignmentScores.iloc[j, i]
        if i % 100 == 0:
            print(
            f"Done with filling {i} rows from {(model).split('/')[1]}'s embeddings comparison matrix"
        )
    embeddingAlignmentScores.to_csv(
        f"Embedding Alignment Scores/Filled/Hyena/{(model).split('/')[1]}Scores.csv" # remove hyena
    )
    finish = time.time()
    return f'Finished filling the embedding alignment scores for {(model).split('/')[1]} in {round(finish - start)} seconds.'

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        results = executor.map(main, models)
        for result in results:
            print(result)
