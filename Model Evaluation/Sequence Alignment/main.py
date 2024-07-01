from Bio import SeqIO, Align
import numpy as np
import pandas as pd
import concurrent.futures
import time

sequences = []


def seqReader(file, fileType):
    count = 1
    for sequence in SeqIO.parse(file, fileType):
        sequences.append({"id": count, "seq": sequence.seq})
        count += 1


seqReader("YAL001C.fasta", "fasta")

aligner = Align.PairwiseAligner()


def align(seq1, seq2):
    alignments = aligner.align(seq1, seq2)
    return alignments.score


def rowAdder(i):
    # print(f"Starting pairwise alignment of sequence {i+1}")
    row = np.zeros(len(sequences))
    row[i] = len(
        sequences[i]["seq"]
    )  # Since aligning a sequence with itself gives a full score
    for j in range(i + 1, len(sequences)):
        seq1, seq2 = sequences[i]["seq"], sequences[j]["seq"]
        row[j] = align(seq1, seq2)

    # print(f"Finished pairwise alignment of sequence {i+1} with all other sequences.")
    return i, row


# alignmentScores = pd.DataFrame(columns=[i for i in range(len(sequences))])
alignmentScores = pd.read_csv("alignmentScores.csv")

if __name__ == "__main__":
    limit = 115
    while limit <= len(sequences):
        start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
            results = executor.map(rowAdder, range(limit - 15, limit))
            for result in results:
                i, row = result
                alignmentScores[i] = row
        finish = time.perf_counter()
        print(
            f"Done with pairwise comparison of {limit-15} to {limit} sequences in {round(finish-start)%60} minutes."
        )
        alignmentScores.to_csv("alignmentScores.csv")
        limit += 100
