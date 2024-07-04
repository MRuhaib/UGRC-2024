from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import pandas as pd
import array

models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]

for model in models:
    name = model.split("/")[1]
    with open(f"Embeddings/{name}.txt", "r+") as f:
        modelDict = eval(f.read())
        print(model, len(modelDict[0]["embedding"]))
