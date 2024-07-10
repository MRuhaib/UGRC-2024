from Bio import SeqIO, Align
import biotite.sequence.align as biotiteAlign
import biotite.sequence.io.fasta as fasta
from biotite.sequence import NucleotideSequence
import numpy as np
import pandas as pd
import concurrent.futures
import time

module = "Biotite"  # else, biopython

sequences = []


def seqReader(file):
    count = 1
    if module == "Biopython":
        for sequence in SeqIO.parse(file, "fasta"):
            sequences.append({"id": count, "seq": sequence.seq})
            count += 1
    elif module == "Biotite":
        fastaFile = fasta.FastaFile.read(file)
        f = fasta.get_sequences(fastaFile)
        for key, value in f.items():
            sequences.append({"id": count, "seq": value})
            count += 1


matrix = biotiteAlign.SubstitutionMatrix.std_nucleotide_matrix()


def align(seq1, seq2):
    if module == "Biopython":
        aligner = Align.PairwiseAligner()
        alignment = aligner.align(seq1, seq2)
        return alignment.score

    elif module == "Biotite":
        alignment = biotiteAlign.align_optimal(seq1, seq2, matrix)
        return alignment[0].score


seqReader("YAL001C.fasta")
alignmentScores = pd.DataFrame(columns=[i for i in range(len(sequences))])


def rowAdder(i):
    print(f"Starting pairwise alignment of sequence {i+1}")
    row = np.zeros(len(sequences))
    row[i] = (
        len(sequences[i]["seq"]) * 5
    )  # Since aligning a sequence with itself gives a full score; with biotite a full score is apparently 5 times the sequence length?
    for j in range(i + 1, len(sequences)):
        seq1, seq2 = sequences[i]["seq"], sequences[j]["seq"]
        row[j] = align(seq1, seq2)
    # print(f"Finished pairwise alignment of sequence {i+1} with all other sequences.")
    with open(f"Sequence Alignment Scores/{module}/rows.txt", "a+") as f:
        f.write(str({"seq": i, "row": list(row)}) + ",\n")
    return i, row


if __name__ == "__main__":
    """
    upperLimit = 20
    lowerLimit = upperLimit - 20
    while upperLimit <= len(sequences):
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            results = executor.map(rowAdder, range(lowerLimit, upperLimit))
            for result in results:
                i, row = result
                # alignmentScores[i] = row

        # alignmentScores.to_csv(f"Sequence Alignment Scores/{module}/alignmentScores.csv", mode="a")
        finish = time.time()
        print(
            f"Done with pairwise comparison of {lowerLimit} to {upperLimit} sequences in {round(finish-start, 2)} seconds."
        )

        if upperLimit == len(sequences) - (len(sequences) % 20):  # i.e., 1000
            upperLimit += 11
            lowerLimit = upperLimit - 11
        else:
            upperLimit += 20
            lowerLimit = upperLimit - 20
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        results = executor.map(rowAdder, [400, 408, 410, 421, 552])
        for result in results:
            i, row = result
    print(
        f"Done with pairwise comparison of {lowerLimit} to {upperLimit} sequences in {round(finish-start, 2)} seconds."
    )

# TODO: refactor all of this into jupiter notebooks
