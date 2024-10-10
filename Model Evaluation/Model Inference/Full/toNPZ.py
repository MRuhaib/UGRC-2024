import numpy as np
import time
import json
from pathlib import Path

start = time.time()

f = Path("Hyena/hyenadna-medium-160k-seqlen-hf.json")
data = json.loads(f.read_text())

print(data[:100])

data = eval(data)

embeddings = data["embeddings"]

finish = time.time()
print("done reading in", finish - start, "seconds")

data = {element["id"]: element["embedding"] for element in embeddings}

np.savez("Hyena/hyenadna-medium-160k-seqlen-hf.npz", **data)
