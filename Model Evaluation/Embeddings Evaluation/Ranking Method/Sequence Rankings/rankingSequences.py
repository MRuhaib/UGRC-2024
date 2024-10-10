import pandas as pd
import numpy as np

scores = pd.read_csv(
    "../../../Sequence Alignment/Sequence Alignment Scores/alignmentScoresFilled.csv"
)
# Note that when reading these csv files, delete the first column from the csv file so that it can be read properly by pandas.
totalSeqs = len(scores)
rankings = []


# create a ranking of the other sequences according to their similarity with the sequence currently being passed
def generateRanking(seq):
    row = scores.loc[seq]
    seqScores = []
    ranking = []
    for i, score in enumerate(row):
        if i != seq:
            seqScores.append(score)
        else:
            seqScores.append(
                10000
            )  # So that the current 'anchor' sequence is not considered in the ranking; it gets popped out of the list later.
    seqScores = np.argsort(
        seqScores
    )  # Since the sequence number is the same as the score's index in the row, this returns a sorted array with the indices corresponding to the sorted scores.
    for index in seqScores:
        ranking.append(index)
    ranking.pop()
    rankings.append({"seq": seq, "ranking": ranking})


for seq in range(totalSeqs):
    generateRanking(seq)

with open("seqRankings.txt", "w+") as f:
    f.write(str(rankings))

print("Finished creating rankings for all the sequences")
