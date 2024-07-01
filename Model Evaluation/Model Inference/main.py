from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch


def generateEmbeddings(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModel.from_pretrained(modelName)

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

    return modelName, embeddings


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
        modelName, embeddings = result
        modelEmbeddings.append({"model": modelName, "embeddings": embeddings})

with open("ModelResults.txt", "w+") as f:
    f.write(str(modelEmbeddings))
