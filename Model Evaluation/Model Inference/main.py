from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from Bio import SeqIO
import numpy as np
import time
import concurrent.futures
import torch

sequences = []

device = "cuda" if torch.cuda.is_available() else "cpu"


def seqReader(file, fileType):
    count = 1
    for sequence in SeqIO.parse(file, fileType):
        sequences.append({"id": count, "seq": sequence.seq})
        count += 1


seqReader("../Sequence Alignment/YAL001C.fasta", "fasta")


def generateEmbeddings(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        modelName,
        trust_remote_code=True,
    ).to(device)
    embeddings = []

    start = time.time()
    print(f"Starting to generate {modelName}'s embeddings.")
    for element in sequences:
        sequence = str(element["seq"])
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = [
            item
            for item in outputs.last_hidden_state.cpu().mean(dim=1).squeeze().numpy()
        ]
        embeddings.append({"seq": element["id"], "embedding": embedding})
    finish = time.time()

    print(
        f"Finished generating {modelName}'s embeddings in {round(finish - start)} seconds."
    )
    with open(f"{modelName.split('/')[1]}.txt", "w+") as f:
        f.write(str(embeddings))


"""
Done with: 

    #"LongSafari/hyenadna-medium-450k-seqlen-hf",
    #"InstaDeepAI/nucleotide-transformer-500m-human-ref",
    #"AIRI-Institute/gena-lm-bigbird-base-t2t", #Max input sequence - 4096
    #"LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",

Not Working:
    #"zhihan1996/DNABERT-S",
    #"zhihan1996/DNABERT-2-117M", - wrong config? The model class you are passing has a `config_class` attribute that is not consistent with the config class you passed
and when using bertconfig: TypeError("dot() got an unexpected keyword argument 'trans_b'")
                           UserWarning: Increasing alibi size from 512 to 709
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
    "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    "ctheodoris/Geneformer", 
    "EleutherAI/enformer-official-rough", - unrecognized architecture?

Not on huggingface: 
    borzoi, satori, scGPT
"""

models = [
    # add models here
]

if __name__ == "__main__":
    for model in models:
        generateEmbeddings(model)

"""
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(generateEmbeddings, models)
    for result in results:
        modelName, embeddings = result
        modelEmbeddings.append({"model": modelName, "embeddings": embeddings})
"""
# TODO: refactor all of this into jupiter notebooks
