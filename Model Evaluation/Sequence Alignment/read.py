from pathlib import Path

directory = "../Model Inference/Genomes"
count = 0
for file in Path(directory).glob("*.fasta"):
    count += 1
    strain = str(file).split("\\")[-1].rstrip(".fasta")
    print(strain)
    if count == 10:
        break
