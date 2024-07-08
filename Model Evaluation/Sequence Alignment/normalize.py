import numpy as np
import torch
import pandas as pd
import array

scores = pd.read_csv("Sequence Alignment Scores/alignmentScoresFilled.csv")


def normalize(scores):
    scores = scores.astype(float)
    seqLen = scores.loc[0][0]
    for i in range(len(scores)):
        scores.loc[i] = scores.loc[i] / seqLen
    scores.to_csv("Sequence Alignment Scores/normalizedScores.csv")


if __name__ == "__main__":
    normalize(scores)
