import numpy as np
import pandas as pd
import biotite.sequence.align as biotiteAlign
import biotite.sequence.io.fasta as fasta
from biotite.sequence import NucleotideSequence

metadata = pd.read_csv("metadata2.csv")


def getUniqueClades():
    clades = metadata["Clades"]
    uniqueClades = []
    distribution = []
    for clade in clades:
        if clade not in uniqueClades:
            uniqueClades.append(clade)

    for clade in uniqueClades:
        count = 0
        for seq in clades:
            if seq == clade:
                count += 1
        distribution.append({"clade": clade, "count": count})

    with open("clades.txt", "w+") as f:
        f.write(str(distribution))


# Note that 'UN' represents 'unclustered'!


def getSeqClades(file):
    sequences = []
    count = 1
    fastaFile = fasta.FastaFile.read(file)
    f = fasta.get_sequences(fastaFile)
    for key, value in f.items():
        name = key.split("_")
        if len(name) == 4:
            name = f"{name[0]}_{name[1]}"
        else:
            name = f"{name[0]}"
        row = metadata.loc[metadata["Standardized name"] == name]
        clade = row["Clades"].iloc[0]
        sequences.append(
            {"id": count, "name": name, "clade": clade.rstrip(" "), "seq": value}
        )
        count += 1

    with open("labels.txt", "w+") as f:
        f.write(str(sequences))


getSeqClades("YAL001C.fasta")
