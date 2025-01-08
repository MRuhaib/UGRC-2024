import numpy as np
import pandas as pd
import time
from pathlib import Path
from Bio import SeqIO
import shutil  # Can copy files from one directory to another!
import os
import json

"""
This script is to find the genes from the Peter database, whose individual lengths are the closest to the fractions of the models' context windows.
For models with larger context windows, different combinations of genes from the Peter database are used so that they come close to the the different fractions of these models' context windows.
For each model, calculate 25, 50, 75 and 100% lengths of its context window - these are the 'fractions'.
Then, find the genes whose lengths are the closest to each of these.
"""
modelLengths = {
    "DNABERT-2": 500,
    "GPN": 500,
    "GROVER": 500,
    "NT": 1000,
    "Hyena-1k": 1000,
    "DNABERT-S": 10000,
    "Hyena-16k": 16000,  # Largest individual gene from the peter database is 14.5k
    "Hyena-32k": 32000,  # Note that for this and the next one, the largest gene doesn't even come to 50% of this.
    "GenaLM": 36000,
    "Hyena-160k": 160000,
    "Hyena-450k": 450000,
    "Hyena-1m": 1000000,
}

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
    "Hyena-160k": {
        "25": ["x", 1000000],
        "50": ["x", 1000000],
        "75": ["x", 1000000],
        "100": ["x", 1000000],
    },
    "Hyena-450k": {
        "25": ["x", 1000000],
        "50": ["x", 1000000],
        "75": ["x", 1000000],
        "100": ["x", 1000000],
    },
    "Hyena-1m": {
        "25": ["x", 1000000],
        "50": ["x", 1000000],
        "75": ["x", 10000000],
        "100": ["x", 10000000],
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


# Finding the genes : assign initial genes for each fraction
def singleGeneFinder():
    for key, value in modelLengths.items():
        model = key
        lengths = {
            "25": int(0.25 * value),
            "50": int(0.5 * value),
            "75": int(0.75 * value),
            "100": value,
        }
        for gene in allGenes:
            geneLength = gene[1]
            for fraction, length in lengths.items():
                if abs(length - geneLength) < abs(
                    closestGenes[model][fraction][1] - length
                ):
                    closestGenes[model][fraction] = gene


def geneCombiner(requiredLength, difference, combinedGenes):
    # Select the largest gene closest to this fraction, get remaining length, get largest gene closest to that, get remaining length, and so on - basically a recursive function!
    # First pass: difference = remaining length between allotted gene and search thru global allGenes list, add the largest to the genesToBeCombined array.
    if difference < 100:
        print(requiredLength, combinedGenes[1])
        return
    else:
        minGene = ["x", 100000000]  # initial placeholder
        genesToBeCombined = combinedGenes[0]
        # totalLength = combinedGenes[1]
        for gene in allGenes:
            if (
                abs(difference - gene[1]) < abs(difference - minGene[1])
                and gene[0] not in genesToBeCombined
            ):
                minGene = gene
        genesToBeCombined.append(minGene[0])
        combinedGenes[1] += minGene[1]
        newDiff = requiredLength - combinedGenes[1]  # totalLength
        geneCombiner(requiredLength, newDiff, combinedGenes)


def geneAllocator():
    for model, fractions in closestGenes.items():
        modelLength = modelLengths[model]
        lengths = {
            "25": int(0.25 * modelLength),
            "50": int(0.5 * modelLength),
            "75": int(0.75 * modelLength),
            "100": modelLength,
        }
        for fraction, length in lengths.items():
            totalLength = fractions[fraction][1]  # Total length of the selected gene(s)
            difference = length - totalLength
            if difference >= 100:
                combinedGenes = [[fractions[fraction][0]], totalLength]
                geneCombiner(length, difference, combinedGenes)
                fractions[fraction] = combinedGenes


# Saving the genes in the appropriate folders:
def saveGenes():
    genesDirectory = "C:/Users/Ruhaib/Downloads/1011CDS_withAmbiguityResidues/All Genes"

    for key, value in closestGenes.items():
        model = key
        for lenType, nearestGene in value.items():
            os.makedirs(
                f"./{model}/{lenType}", exist_ok=True
            )  # os.makedirs handles the parent directories that aren't created yet!
            seqFileName = ""
            if isinstance(nearestGene[0], str):
                gene = nearestGene[0]
                seqFileName = gene
                seq = seqReader(f"{genesDirectory}/{gene}.fasta")
            elif isinstance(nearestGene[0], list):
                genes = nearestGene[0]
                seqFileName = f"{model}_{lenType}_combined"
                seq = ""
                for gene in genes:
                    seq += seqReader(f"{genesDirectory}/{gene}.fasta")
            with open(f"./{model}/{lenType}/{seqFileName}.txt", "w+") as f:
                f.write(seq)
            # shutil.copy(f"{genesDirectory}/{gene}.fasta", f"./{model}/{lenType}")


singleGeneFinder()
geneAllocator()
saveGenes()

with open("closestGenes.json", "w+") as f:
    json.dump(closestGenes, f)
