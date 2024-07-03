import torch

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
    with open(f"../Model Inference/Embeddings/{name}.txt", "r+") as f:
        arr = f.read()
        print(len(arr), arr[:100])

test = torch.tensor([1, 2, 3])
with open("test.txt", "w+") as f:
    f.write(test)
