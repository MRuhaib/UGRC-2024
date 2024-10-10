from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time
import json
from Bio import SeqIO
from pathlib import Path
from more_itertools import batched

hyenaModels = [
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    "LongSafari/hyenadna-small-32k-seqlen-hf",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]

group = "DNABERT"  #'Hyena' #else, 'General'

finishedModels = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]

DNABERTModels = [
    "zhihan1996/DNABERT-S",
    "zhihan1996/DNABERT-2-117M",
    "zhihan1996/DNA_bert_6",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


def generateEmbeddings(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        modelName,
        trust_remote_code=True,
    ).to(device)

    embeddings = {}
    # embeddingsSum = []
    # embeddingsAvg = []

    start = time.time()

    finishedSequences = []

    directory = "Genomes"
    maxLen = 10000  # For DNABERT-S

    count = 0
    embeddingsFile = Path(f"Full/{group}/{modelName.split('/')[1]}_{maxLen}.npz")

    if embeddingsFile.is_file():
        # data = eval(json.loads(embeddingsFile.read_text()))
        # embeddings = data['embeddings']
        data = np.load(f"Full/{group}/{modelName.split('/')[1]}_{maxLen}.npz")
        strains = data.files
        embeddings = {strain: data[strain] for strain in strains}
        count = len(strains)
        for strain in strains:
            finishedSequences.append(directory + "/" + strain + ".fasta")
        print(len(finishedSequences), "sequences done so far.")

    print(
        f"Starting to generate {modelName}'s embeddings with sequence number", count + 1
    )

    for filename in Path(directory).glob("*.fasta"):
        if str(filename) not in finishedSequences:
            count += 1
            genome = SeqIO.read(filename, "fasta")
            sequence = str(genome.seq)
            # maxLen = int(tokenizer.model_max_length) - 2 #Subtracting just in case some errors occur
            segmentedSequences = [
                sequence[i : i + maxLen] for i in range(0, len(sequence), maxLen)
            ]
            batchedSequences = list(batched(segmentedSequences, 3))
            """
            Hyena:
            1k models - batch size of 1600
            160k model - batch size of 4
            450k model - batch size of 3, while halving the maxlen = 215000 (cause otherwise it takes up 17 gb ram even without batching)
            1m model - batch size of 1, while having maxlen = 666667; the max the gpu would allow

            DNABERT-S: batch size of 10, with maxLen = 10000, since that's the sequence length the model was trained on.
            """

            id = str(genome.id)

            seqEmbedding = []

            for batch in batchedSequences:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # result = outputs.last_hidden_state.cpu().mean(dim=1).squeeze().numpy()
                result = outputs[1].cpu().numpy().squeeze()  # For DNABERT-S

                if result.ndim == 1:
                    embedding = [item for item in result]
                    seqEmbedding.append(embedding)
                else:  # since the embeddings are batched here
                    for element in result:
                        embedding = [item for item in element]
                        seqEmbedding.append(embedding)

            embeddings[id] = seqEmbedding
            # embeddings.append({"id": id, "embedding":seqEmbedding})
            # embeddingsSum.append({"id": id, "embedding": [item for item in np.sum(np.array(seqEmbedding), axis = 0)]})
            # embeddingsAvg.append({"id": id, "embedding": [item for item in np.sum(np.array(seqEmbedding), axis = 0)/len(seqEmbedding)]})

            np.savez_compressed(
                f"Full/{group}/{modelName.split('/')[1]}_{maxLen}", **embeddings
            )
            """
            if count % 100 == 0:
                path = f"Full/{group}/{modelName.split('/')[1]}_{maxLen}_{int(count/100)}"

            np.savez_compressed(path)
            """

            if count % 10 == 0:
                finish = time.time()

                print(
                    f"Done with {count} sequences in {round(finish - start)} seconds."
                )
                """
                data = {
                    "embeddings": embeddings,
                    #"embeddingsAvg": embeddingsAvg,
                    #"embeddingsSum": embeddingsSum,
                }

                with open(f"Full/{group}/{modelName.split('/')[1]}5k.json", "w+") as f:
                    json.dump(str(data), f, indent=4)
                """
            elif count == 1011:
                finish = time.time()
                """
                with open(f"Full/{group}/{modelName.split('/')[1]}5k.json", "w+") as f:
                    json.dump(str(data), f, indent=4)
                """
                print(
                    f"Finished generating {modelName}'s embeddings in {round(finish - start)} seconds."
                )
        else:
            continue


"""
for model in hyenaModels:
    generateEmbeddings(model)
"""


def recursiveExecutor():
    try:
        generateEmbeddings(DNABERTModels[0])
    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        time.sleep(60)
        print("Restarting...")
        recursiveExecutor()


if __name__ == "__main__":
    # recursiveExecutor()
    generateEmbeddings(DNABERTModels[0])
