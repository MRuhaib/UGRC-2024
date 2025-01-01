import numpy as np
import pandas as pd
import random
import time
import os
from scipy.spatial.distance import hamming
from pathlib import Path
from Bio import SeqIO

types = ["Coding", "Non-coding", "Mixed"]
numMutations = 100  # Number of mutations for a specific value of x
fractions = ["25", "50", "75", "100"]

models = [
    "DNABERT-2",
    "GPN",
    "GROVER",
    "NT",
    "DNABERT-S",
    "GenaLM",
    "Hyena-1k",
    "Hyena-16k",
    "Hyena-32k",
]


def mutater(x, seq):
    seqLength = len(seq)
    mutationNumber = round(
        x * seqLength / 100
    )  # Total number of bases that will be replaced i.e., mutated in the original sequence.
    alreadyMutated = []
    for N in range(mutationNumber):
        i = random.randrange(
            seqLength
        )  # Randomly chosen position at which the base will be replaced.
        while (
            i in alreadyMutated
        ):  # A previously mutated base mustn't be mutated again!
            i = random.randrange(seqLength)

        alreadyMutated.append(i)

        ogBase = seq[i]
        newBase = random.choice(
            ["A", "T", "C", "G"]
        )  # Replacement base is also randomly chosen.
        while newBase == ogBase:  # Must be different from the original base, obviously.
            newBase = random.choice(["A", "T", "C", "G"])
        seq = seq[:i] + newBase + seq[i + 1 :]
    return seq
    # Save the new sequence in Mutated Sequences/(Sequence Type)/(Length)


def seqReader(file):
    for sequence in SeqIO.parse(file, "fasta"):
        seq = sequence.seq
        break  # taking only the first strain's sequence.
    return seq


if __name__ == "__main__":
    seqType = types[0]  # change this manually
    start = time.time()
    for model in models:
        print("Now starting with:", model)
        for fraction in fractions:
            mainPath = f"{seqType}/{model}/{fraction}"
            ogDirectory = f"./Original Sequences/{mainPath}"
            newDirectory = f"./Mutated Sequences/{mainPath}"
            os.makedirs(f"{newDirectory}", exist_ok=True)
            for filename in Path(ogDirectory).glob("*.txt"):
                ogSeq = ""
                with open(filename, "r+") as file:
                    ogSeq = str(file.read())
                gene = str(filename).split("\\")[-1].rstrip(".txt")
                for x in range(5, 51, 5):
                    mutations = []
                    for i in range(numMutations):
                        mutSeq = mutater(x, ogSeq)
                        mutations.append(str(mutSeq))

                    with open(f"{newDirectory}/{gene}_{x}.txt", "w+") as f:
                        f.write(str(mutations))
            oneFractionDone = time.time()
            print(
                f"Done with fraction {fraction} for model {model} in {round(oneFractionDone - start, 2)} seconds."
            )


"""
#Verification run: it works :))
if __name__ == "__main__":
    gene = "YLR106C"
    sequences = seqReader(f"Original Sequences/{seqType[0]}/10K+/{gene}.fasta")
    sampleSeq = sequences[0]["seq"]
    percentage = 70
    newSeq = mutater(percentage, sampleSeq)
    x = hamming(newSeq, sampleSeq)
    print("Percentage of the sequence mutated is:", round(x * 100))
"""
