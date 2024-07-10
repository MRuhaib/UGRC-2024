import pandas as pd
from Bio import SeqIO
import time

sequences = []

module = "Biotite"


def seqReader(file, fileType):
    count = 1
    for sequence in SeqIO.parse(file, fileType):
        sequences.append({"id": count, "seq": sequence.seq})
        count += 1


seqReader("YAL001C.fasta", "fasta")

alignmentScores = pd.read_csv(f"Sequence Alignment Scores/{module}/alignmentScores.csv")
# Note that when reading these csv files, delete the first column from the csv file so that it can be read properly by pandas.

start = time.time()

for i in range(len(sequences)):
    for j in range(len(sequences)):
        alignmentScores.iloc[i, j] = alignmentScores.iloc[j, i]

alignmentScores.to_csv(f"Sequence Alignment Scores/{module}/alignmentScoresFilled.csv")


finish = time.time()

print(f"Done with filling dataframe in {round(finish - start) % 60} minutes.")
