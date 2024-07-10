import biotite.sequence.align as biotiteAlign
import biotite.sequence.io.fasta as fasta
from biotite.sequence import NucleotideSequence
import time

fastaFile = fasta.FastaFile.read("YAL001C.fasta")

"""
count = 1
for header, string in fastaFile.items():
    sequences.append({"id": count, "seq": string})
    count += 1
"""
sequences = []

file = fasta.get_sequences(fastaFile)

for key, value in file.items():
    sequences.append(value)

print(len(sequences))

scores = []

start = time.time()
matrix = biotiteAlign.SubstitutionMatrix.std_nucleotide_matrix()

for i in range(len(sequences)):
    alignments = biotiteAlign.align_optimal(
        sequences[0],
        sequences[i],
        matrix,
    )
    print(alignments[0].score)
    scores.append(alignments[0].score)
finish = time.time()
print(f"finished in {finish - start} seconds.")
