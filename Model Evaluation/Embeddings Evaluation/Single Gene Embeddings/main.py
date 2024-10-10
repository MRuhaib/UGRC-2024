import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import concurrent.futures

"""
Implement both methods here - direct score comparison and rankings comparison
Method 1: rankings evaluation: use the ndcg metric.
Method 2: direct similarity score comparison: normalize seq alignment scores by dividing by 3573 and calculate each model's cos sim score loss against it; then aggregate it all.
"""
seqAlignmentMethod = 'Biopython' #else, biopython
distMethod = 'Euclidean' #else, use Cosine/Filled

hyenaModels = [
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    "LongSafari/hyenadna-small-32k-seqlen-hf",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]

models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    "zhihan1996/DNABERT-S",
    "zhihan1996/DNABERT-2-117M",
    "songlab/gpn-brassicales",
    "PoetschLab/GROVER",
]

genes = [
    ["YAL001C", 3573],
    ["YAL007C", 648],
    ["YAL018C", 978],
    ["YAL022C", 1554],
    ["YAL026C", 4068],
    ["YJR136C", 1266],
    ["YPR192W", 918],
]

#Method 2:

def evaluateScoreLoss(model): #Defunct; ignore this.
    modelScores = pd.read_csv(f'Model Rankings/Embeddings Alignment/Embedding Alignment Scores/{distMethod}/Hyena/{model.split('/')[1]}Scores.csv') #remove hyena
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

def evaluateScoreCorr(model, gene):
    seqScores = pd.read_csv(f"../Sequence Alignment/Sequence Alignment Scores/{seqAlignmentMethod}/Genes/{gene}_scores.csv")
    modelScores = pd.read_csv(f"C:/Users/Ruhaib/Downloads/IIT Stuff/Research/Nirav Sir's Lab/Model Evaluation/Embeddings Evaluation/Single Gene Embeddings/Embeddings Alignment/Embedding Alignment Scores/{distMethod}/{gene}/{model.split('/')[1]}_{gene}_distances.csv") #remove hyena
    corr = modelScores.corrwith(seqScores)
    return corr, model

def evaluateFlattenedCorr(model, gene):
    seqScores = pd.read_csv(f"C:/Users/Ruhaib/Downloads/IIT Stuff/Research/Nirav Sir's Lab/Model Evaluation/Sequence Alignment/Sequence Alignment Scores/{seqAlignmentMethod}/Genes/{gene}_scores.csv").to_numpy().flatten()
    modelScores = pd.read_csv(f"C:/Users/Ruhaib/Downloads/IIT Stuff/Research/Nirav Sir's Lab/Model Evaluation/Embeddings Evaluation/Single Gene Embeddings/Embeddings Alignment/Embedding Alignment Scores/{distMethod}/{gene}/{model.split('/')[1]}_{gene}_distances.csv").to_numpy().flatten()
    return stats.pearsonr(seqScores, modelScores)[0]


if __name__ == "__main__":
    print(f"Using {seqAlignmentMethod}'s sequence alignment scores and {distMethod} metric for embedding comparison:")

    '''
    #For displaying all correlations of each model:
    for gene in genes:
        plt.rcParams['font.size'] = 6
        fig, axs = plt.subplots(2, 4)
        count = 0
        for model in models:
            corr, model = evaluateScoreCorr(model, gene[0])
            print(f"Average correlation of model {model.split('/')[1]}'s embeddings for each sequence is: {sum(corr)/len(corr)}.")
            axs[count//4][count%4].hist(corr, bins = 50)
            axs[count//4][count%4].set_title(f'{model.split('/')[1]}')
            axs[count//4][count%4].set(xlim = (-1, 1), xlabel = "Correlation", ylabel = "Number of Strains")
            count += 1
        fig.tight_layout(pad = 2)
        fig.suptitle(f"{gene[0]} gene; {gene[1]} nucleotides\nCorrelation between the models' embeddings' pairwise euclidean distances and the strain sequences' pairwise sequence alignment scores", fontsize=8, y=1)
        #plt.subplots_adjust(bottom = 1)
        plt.show()
        fig.savefig(f"C:/Users/Ruhaib/Downloads/IIT Stuff/Research/Nirav Sir's Lab/Model Evaluation/Embeddings Evaluation/Single Gene Embeddings/Histograms/Euclidean/Individual/{gene[0]}_biopython.png")
    '''

    for gene in genes:
        x, y = [], []
        fig, ax = plt.subplots()
        for model in models:
            name = model.split('/')[1]
            y.append(evaluateFlattenedCorr(model, gene[0]))
            x.append(name.split('-')[0] + ' ' + name.split('-')[1] + '\n' + name.split('-')[2] + ' ' + name.split('-')[3] if len(name.split('-')) > 3 else name)

        ax.set_title(f"{gene[0]} gene; {gene[1]} nucleotides\nCorrelation between the models' embeddings' pairwise euclidean distances and the strain sequences' pairwise sequence alignment scores", fontsize=10)
        ax.stem(x, y)
        ax.set(ylim=(-1, 0))
        plt.ylabel('Overall correlation between sequences\' Needleman-Wunsch Alignment Scores and embeddings\' Euclidean Distances', fontsize = 8)
        plt.xlabel('Genomic Language Model', fontsize = 8)
        plt.xticks(fontsize = 7)
        plt.show()
        fig.savefig(f"C:/Users/Ruhaib/Downloads/IIT Stuff/Research/Nirav Sir's Lab/Model Evaluation/Embeddings Evaluation/Single Gene Embeddings/Histograms/Euclidean/Overall/{gene[0]}_biopython.png")
        

    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #results = executor.map(evaluateScoreLoss, models)
        results = executor.map(evaluateScoreCorr, models)
        count = 0
        #x, y = [], []
        for result in results:    
            corr, model = result
            print(f"Average correlation of model {model.split('/')[1]}'s embeddings for each sequence is: {sum(corr)/len(corr)}.")
            #y.append(sum(corr)/len(corr))
            #x.append(model.split('/')[1])
            
            #for evaluateScoreCorr
            axs[count].hist(corr, bins = 50)
            axs[count].set_title(f'{model.split('/')[1]}')
            axs[count].set(ylim=(0, 600), xlim = (-1, 1))
            
            #for evaluateScoreLoss:
            # performance, totalLoss, model, lossDF = result
            #x, y = [], []
            #y.append(totalLoss)
            #x.append(model.split('/')[1])
            # STD = lossDF.std().std()
            # axs[count].hist(performance, bins = 20)
            # axs[count].set_title(f'{model.split('/')[1]}')
            # #print(f"Total loss of model {model.split('/')[1]}  is: {totalLoss}.")
            # print(f"Total STD of {model.split('/')[1]}  is: {STD}")
            # with open(f"Model Performance/Hyena/{model.split('/')[1]}.txt", "w+") as f: #remove hyena
            #     f.write(str(performance))
            
            count += 1

    fig, ax = plt.subplots()
    ax.stem(x, y)
    ax.set(ylim=(0, 1))
    
    plt.show()
    '''
