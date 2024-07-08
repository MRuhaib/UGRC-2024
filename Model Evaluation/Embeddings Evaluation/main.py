import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import concurrent.futures

"""
Implement both methods here - direct score comparison and rankings comparison
Method 1: rankings evaluation: use the ndcg metric.
Method 2: direct similarity score comparison: normalize seq alignment scores by dividing by 3573 and calculate each model's cos sim score loss against it; then aggregate it all.
"""

"""
models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]
"""
models = [
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    "LongSafari/hyenadna-small-32k-seqlen-hf",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]
seqScores = pd.read_csv(
    "../Sequence Alignment/Sequence Alignment Scores/normalizedScores.csv"
)

#Method 2:

def evaluateScores(model):
    modelScores = pd.read_csv(f'Model Rankings/Embeddings Alignment/Embedding Alignment Scores/Filled/Hyena/{model.split('/')[1]}Scores.csv') #remove hyena
    lossDF = pd.DataFrame(columns=[i for i in range(len(modelScores))])
    performance = []
    totalLoss = 0
    for seq in range(len(modelScores)):
        trueScore = seqScores.iloc[seq]
        modelScore = modelScores.iloc[seq]
        loss = abs(trueScore - modelScore) #Since higher cos-sim score = higher sequence alignment score = higher similarity
        lossDF[seq] = loss
        #loss = mean_absolute_error(trueScore, modelScore)
        totalLoss += loss
        performance.append(sum(loss))
    #lossDF.std.to_csv(f"Model Performance/Standard Deviations/Hyena/{model.split('/')[1]}.csv") #remove hyena
    totalLoss = round(totalLoss, 3)
    return performance, totalLoss, model, lossDF

if __name__ == "__main__":
    plt.rcParams['font.size'] = 6
    fig, axs = plt.subplots(1, 7)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(evaluateScores, models)
        count = 0
        for result in results:    
            performance, totalLoss, model, lossDF = result
            #x, y = [], []
            #y.append(totalLoss)
            #x.append(model.split('/')[1])
            STD = lossDF.std().std()
            axs[count].hist(performance, bins = 20)
            axs[count].set_title(f'{model.split('/')[1]}')
            #print(f"Total loss of model {model.split('/')[1]}  is: {totalLoss}.")
            print(f"Total STD of {model.split('/')[1]}  is: {STD}")
            '''
            with open(f"Model Performance/Hyena/{model.split('/')[1]}.txt", "w+") as f: #remove hyena
                f.write(str(performance))
            '''
            count += 1
    plt.show()
    
    #fig, ax = plt.subplots()
    #ax.stem(x, y)
    #ax.set(ylim=(0, 0.001))
    
