from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import numpy as np
import torch
import time

sequences = []


def seqReader(file, fileType):
    count = 1
    for sequence in SeqIO.parse(file, fileType):
        sequences.append({"id": count, "seq": sequence.seq})
        count += 1


seqReader("YAL001C.fasta", "fasta")


def generateEmbeddings(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModel.from_pretrained(modelName)
    start = time.perf_counter()
    embeddings = []
    for element in sequences:
        sequence = element["seq"]
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            output_hidden_states=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    finish = time.perf_counter()

    return modelName, embeddings, start, finish


models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bert-base-t2t",
    "zhihan1996/DNABERT-S",
    "zhihan1996/DNABERT-2-117M",
    "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    "EleutherAI/enformer-official-rough",
]

import concurrent.futures

modelEmbeddings = []

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(generateEmbeddings, models)
    for result in results:
        modelName, embeddings, start, finish = result
        modelEmbeddings.append({"model": modelName, "embeddings": embeddings})
        print(
            f"Finished generating embeddings using {modelName} in {round(finish - start) % 60} minutes."
        )

with open("ModelResults.txt", "w+") as f:
    f.write(str(modelEmbeddings))
