import numpy as np
import pandas as pd
import time
from pathlib import Path

module = "Biopython"


def normalize(filePath):
    scores = scores = pd.read_csv(filePath).astype(float)
    start = time.time()
    seqLen = scores.loc[0][0]
    for i in range(len(scores)):
        scores.loc[i] = scores.loc[i] / seqLen
    scores.to_csv(filePath, index=False)
    finish = time.time()
    print(
        f"Done with {str(filePath).split('/')[-1].rstrip('.fasta')}'s scores in {round(finish - start)} seconds."
    )


if __name__ == "__main__":
    beginning = time.time()
    count = 0

    directory = f"Sequence Alignment Scores/{module}/Genes"  # change this depending on model directory.

    for file in Path(directory).glob("*.csv"):
        count += 1
        normalize(file)

    end = time.time()
    print(
        f"Done with normalizing {count} dataframes in {round(end - beginning)} seconds."
    )
