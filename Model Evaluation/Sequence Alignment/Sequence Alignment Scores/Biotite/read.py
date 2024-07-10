import numpy as np
import pandas as pd


alignmentScores = pd.DataFrame(columns=[i for i in range(1011)])

rows = []
with open("rows.txt", "r+") as f:
    rows = eval(f.read())
    print(type(rows))


for element in rows:
    alignmentScores[element["seq"]] = element["row"]

alignmentScores.to_csv("alignmentScores.csv")
