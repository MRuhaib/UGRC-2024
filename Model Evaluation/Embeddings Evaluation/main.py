import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures

"""
Implement both methods here - direct score comparison and rankings comparison
Method 1: rankings evaluation: use the ndcg metric.
Method 2: direct similarity score comparison: normalize seq alignment scores by dividing by 3573 and calculate each model's cos sim score loss against it; then aggregate it all.
"""
models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]

seqScores = pd.read_csv(
    "../Sequence Alignment/Sequence Alignment Scores/normalizedScores.csv"
)

#Method 2:

def evaluateScores(model):
    modelScores = pd.read_csv(f'Model Rankings/Embeddings Alignment/Embedding Alignment Scores/Filled/{model.split('/')[1]}Scores.csv')
    performance = []
    totalLoss = 0
    for seq in range(len(modelScores)):
        trueScore = seqScores.iloc[seq]
        modelScore = modelScores.iloc[seq]
        loss = sum(abs(trueScore - modelScore))
        totalLoss += loss
        performance.append({'seq': seq, "loss": loss})
    totalLoss = round(totalLoss/len(modelScores), 3)
    return performance, totalLoss, model

if __name__ == "__main__":
    plt.rcParams['font.size'] = 8
    x, y = [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(evaluateScores, models)
        for result in results:
            performance, totalLoss, model = result
            y.append(totalLoss)
            x.append(model.split('/')[1])
            print(f"Total loss of model {model.split('/')[1]}  is: {totalLoss}")
            with open(f"Model Performance/{model.split('/')[1]}.txt", "w+") as f:
                f.write(str(performance))
    fig, ax = plt.subplots()
    ax.stem(x, y)
    ax.set(ylim=(0, 25))
    plt.title('Embedding cos-sim score against actual sequence similarity score')
    plt.xlabel('Models')
    plt.ylabel('Total Loss')
    plt.show()


