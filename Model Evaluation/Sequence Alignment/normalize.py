import numpy as np
import pandas as pd
import array

module = "Biotite"
scores = pd.read_csv(f"Sequence Alignment Scores/{module}/alignmentScoresFilled.csv")


def normalize(scores):
    scores = scores.astype(float)
    seqLen = scores.loc[0][0]
    for i in range(len(scores)):
        scores.loc[i] = scores.loc[i] / seqLen
    scores.to_csv(f"Sequence Alignment Scores/{module}/normalizedScores.csv")


if __name__ == "__main__":
    normalize(scores)
