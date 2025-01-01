import numpy as np
import pandas as pd
import time
from pathlib import Path
from Bio import SeqIO
import shutil  # Can copy files from one directory to another!
import os

modelLengths = {
    "DNABERT-2": 500,
    "GPN": 500,
    "GROVER": 500,
    "NT": 1000,
    "Hyena-1k": 1000,
    "DNABERT-S": 10000,
    "Hyena-16k": 16000,
    "Hyena-32k": 32000,  # Note that for this and the next one, the largest gene doesn't even come to 50% of this.
    "GenaLM": 36000,
    # Larger Hyena models have much larger context windows; as such I suupose it's irrelevant to search for gene sequences that match these.
}

"""
For each model, calculate 25, 50, 75 and 100% lengths of its context window
Then, find the genes whose lengths are the closest to each of these.
"""


closestGenes = {  # Stores the genes whose lengths are closest to each of these fractions; initialised with some random initial placeholders
    "DNABERT-2": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
    "GPN": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
    "GROVER": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
    "NT": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
    "Hyena-1k": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
    "DNABERT-S": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
    "Hyena-16k": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
    "Hyena-32k": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
    "GenaLM": {
        "25": ["x", 100000],
        "50": ["x", 100000],
        "75": ["x", 100000],
        "100": ["x", 100000],
    },
}


def seqReader(file):
    for sequence in SeqIO.parse(file, "fasta"):
        seq = sequence.seq
        break  # taking only the first strain's sequence.
    return str(seq)


allGenes = []
with open("sortedGenes.txt", "r+") as f:
    allGenes = eval(f.read())

for key, value in modelLengths.items():
    model = key
    len25 = int(0.25 * value)
    len50 = int(0.5 * value)
    len75 = int(0.75 * value)
    len100 = value
    print(model, len25, len50, len75, len100)
    for gene in allGenes:
        geneLength = gene[1]
        if abs(len25 - geneLength) < abs(closestGenes[model]["25"][1] - len25):
            closestGenes[model]["25"] = gene
        if abs(len50 - geneLength) < abs(closestGenes[model]["50"][1] - len50):
            closestGenes[model]["50"] = gene
        if abs(len75 - geneLength) < abs(closestGenes[model]["75"][1] - len75):
            closestGenes[model]["75"] = gene
        if abs(len100 - geneLength) < abs(closestGenes[model]["100"][1] - len100):
            closestGenes[model]["100"] = gene

genesDirectory = "C:/Users/Ruhaib/Downloads/1011CDS_withAmbiguityResidues/All Genes"

for key, value in closestGenes.items():
    model = key
    for lenType, nearestGene in value.items():
        os.makedirs(
            f"./{model}/{lenType}", exist_ok=True
        )  # os.makedirs handles the parent directories that aren't created yet!
        gene = nearestGene[0]
        firstSeq = seqReader(f"{genesDirectory}/{gene}.fasta")
        with open(f"./{model}/{lenType}/{gene}.txt", "w+") as f:
            f.write(firstSeq)
        # shutil.copy(f"{genesDirectory}/{gene}.fasta", f"./{model}/{lenType}")

with open("closestGenes.txt", "w+") as f:
    f.write(str(closestGenes))
