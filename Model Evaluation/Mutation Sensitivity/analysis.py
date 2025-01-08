"""
1 and 2 will be done separately for each model, whereas 3 is a pan-model comparison.
1. Mutation percentage(x)-wise:
Compare average cosine/euclidean distance between each of the 100 mutated sequence’s embedding with the original embedding, and plot a histogram of the scores - do this for each of the mutation percentages
2. Context window/sequence length percentage-wise:
Average all the distances for a specific mutation; compare it over the 4 types of input sequences for a model
3. Model-wise:
Perform some aggregation on all the above data specific to each model, and then do a comparative analysis of the models. 

"""
