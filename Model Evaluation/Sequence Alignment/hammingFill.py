import pandas as pd
from Bio import SeqIO
import time
from pathlib import Path
import concurrent.futures


def fill(filePath):
    scores = pd.read_csv(filePath)
    start = time.time()

    for i in range(len(scores)):
        for j in range(len(scores)):
            scores.iloc[i, j] = scores.iloc[j, i]

    scores.to_csv(filePath, index=False)

    finish = time.time()
    print(
        f"Done with {str(filePath).split('/')[-1].rstrip('_10000_hammingScores.csv')}'s scores in {round(finish - start)} seconds."
    )


if __name__ == "__main__":
    beginning = time.time()
    count = 0
    csvFiles = []

    directory = "Hamming Scores/DNABERT-S"  # change this depending on model directory.

    for file in Path(directory).glob("*.csv"):
        csvFiles.append(file)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            results = executor.map(fill, csvFiles)
            for result in results:
                count += 1
                end = time.time()
                if count % 10 == 0:
                    print(
                        f"Done with {count} files in {round(end - beginning)} seconds."
                    )

    except Exception as e:
        print(f"Error: {e}")

    print(f"Done with filling {count} dataframes in {round(end - beginning)} seconds.")
