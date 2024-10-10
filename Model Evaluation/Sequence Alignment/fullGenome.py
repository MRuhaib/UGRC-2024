import numpy as np
import pandas as pd
from Bio import SeqIO, Align
import biotite.sequence.align as biotiteAlign
import biotite.sequence.io.fasta as fasta
from biotite.sequence import NucleotideSequence
import time
import concurrent.futures
from scipy.spatial.distance import hamming
from pathlib import Path

module = "Biopython"


def seqSegmenter(file, maxLen):
    if module == "Biopython":
        genome = SeqIO.read(file, "fasta")
        sequence = str(genome.seq)
    elif module == "Biotite":
        fastaFile = fasta.FastaFile.read(file)
        sequence = fasta.get_sequence(fastaFile)
    return [sequence[i : i + maxLen] for i in range(0, len(sequence), maxLen)]


matrix = biotiteAlign.SubstitutionMatrix.std_nucleotide_matrix()


def align(seq1, seq2):
    if module == "Biopython":
        aligner = Align.PairwiseAligner()
        alignment = aligner.align(seq1, seq2)
        return alignment.score

    elif module == "Biotite":
        alignment = biotiteAlign.align_optimal(seq1, seq2, matrix)
        return alignment[0].score


def hammingAlign(seq1, seq2):
    seq1, seq2 = list(seq1), list(seq2)
    return hamming(seq1, seq2)


def main(strain):
    start = time.time()
    maxLen = 215000
    print(f"Starting with strain {strain}")

    segmentedSequences = seqSegmenter(
        f"../Model Inference/Genomes/{strain}.fasta", maxLen
    )
    segmentedSequences.pop()  # to remove the last part that's of a smaller length; cause all the other ones will be of length 160k

    hammingScores = pd.DataFrame(columns=[i for i in range(len(segmentedSequences))])

    for i in range(len(segmentedSequences)):
        row = np.zeros(len(segmentedSequences))
        for j in range(i, len(segmentedSequences)):
            seq1, seq2 = segmentedSequences[i], segmentedSequences[j]
            row[j] = hammingAlign(seq1, seq2)
        hammingScores[i] = row

    hammingScores.to_csv(
        f"Hamming Scores/hyena450/{strain}_{maxLen}_hammingScores.csv", index=False
    )  # manually change this path

    finish = time.time()

    return f"Done with strain {strain} in {round(finish - start)} seconds."


if __name__ == "__main__":
    beginning = time.time()

    directory = "../Model Inference/Genomes"
    strains = []
    count = 0

    for file in Path(directory).glob("*.fasta"):
        strain = str(file).split("/")[-1].rstrip(".fasta")
        strains.append(strain)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
            results = executor.map(main, strains)
            for result in results:
                count += 1
                print("Strain number:", count, result)

    except Exception as e:
        print(f"Error: {e}")

    end = time.time()
    print(
        f"Done with the pairwise Hamming scoring(?) of each strain's segments in {round(end - beginning)} seconds."
    )

"""
def rowAdderHamming(i):
    start = time.time()
    print(f"Starting pairwise alignment of segment {i+1}")
    row = np.zeros(len(segmentedSequences))
    row[i] = (
        maxLen if module == "Biopython" else maxLen * 5
    )  # Since aligning a sequence with itself gives a full score; with biotite a full score is apparently 5 times the sequence length? And the sequence length itself if using biopython
    for j in range(i + 1, len(segmentedSequences)):
        seq1, seq2 = segmentedSequences[i], segmentedSequences[j]
        row[j] = hammingAlign(seq1, seq2)
        finish = time.time()
    finish = time.time()
    print(
        f"Finished pairwise alignment of segment {i+1} with all other segment in {round(finish - start)} seconds."
    )
    with open(f"Alignment Scores/{maxLen}AAB_rows.txt", "a+") as f: #add the module used in the file name too later
        f.write(str({"seq": i, "row": list(row)}) + ",\n")
    return i, row


module = "Biopython"
maxLen = 160000

segmentedSequences = seqSegmenter("../Model Inference/Genomes/AAB.fasta", maxLen)

hammingScores = pd.DataFrame(columns=[i for i in range(len(segmentedSequences))])

if __name__ == "__main__":
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            results = executor.map(rowAdderHamming, range(len(segmentedSequences)))
            for result in results:
                i, row = result
                hammingScores[i] = row

        hammingScores.to_csv(
            f"Alignment Scores/Segments/FullGenome_{maxLen}AAB_hammingScores.csv"
        )
    except Exception as e:
        print(f"Error: {e}")
        hammingScores.to_csv(
            f"Alignment Scores/Segments/FullGenome_{maxLen}AAB_hammingScores.csv"
        )
"""
