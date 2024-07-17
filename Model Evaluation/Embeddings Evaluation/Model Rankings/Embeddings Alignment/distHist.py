import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import concurrent.futures

models = [
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]
seqAlignmentMethod = "Biotite"  # else, biopython
distMethod = "Euclidean"  # else, use Cosine/Filled


def main(model):
    model = model.split("/")[1]
    distances = pd.read_csv(f"Embedding Alignment Scores/Euclidean/{model}.csv")
    print(
        f"Using {seqAlignmentMethod}'s sequence alignment scores and {distMethod} metric for embedding comparison:"
    )
    plt.rcParams["font.size"] = 6
    fig, ax = plt.subplots(figsize=(10, 10))
    # x, y = [], []

    # y.append(sum(corr)/len(corr))
    # x.append(model.split('/')[1])

    # for evaluateScoreCorr
    ax.hist(corr, bins=50)
    ax.set_title(f"{model}")
    ax.set(ylim=(0, 600), xlim=(0, 1))
    """
    #for evaluateScoreLoss:
    performance, totalLoss, model, lossDF = result
    #x, y = [], []
    #y.append(totalLoss)
    #x.append(model.split('/')[1])
    STD = lossDF.std().std()
    axs[count].hist(performance, bins = 20)
    axs[count].set_title(f'{model.split('/')[1]}')
    #print(f"Total loss of model {model.split('/')[1]}  is: {totalLoss}.")
    print(f"Total STD of {model.split('/')[1]}  is: {STD}")
    with open(f"Model Performance/Hyena/{model.split('/')[1]}.txt", "w+") as f: #remove hyena
        f.write(str(performance))
    """


# fig, ax = plt.subplots()
# ax.stem(x, y)
# ax.set(ylim=(0, 1))

plt.show()
