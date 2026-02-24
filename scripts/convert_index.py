# convert_index.py
import pickle, faiss, numpy as np, json, sys

pkl_path = "/data3/bianhaiyang/data/OpenResearcher/OpenResearcher-Indexes/qwen3-embedding-8b/corpus.pkl"
out_faiss = pkl_path.replace(".pkl", ".faiss")
out_lookup = pkl_path.replace(".pkl", "_lookup.json")

print("Loading pickle (this takes a while, last time)...")
with open(pkl_path, "rb") as f:
    reps, lookup = pickle.load(f)

reps = np.array(reps, dtype=np.float32)
print(f"Vectors: {reps.shape}, building FAISS index...")

index = faiss.IndexFlatIP(reps.shape[1])
index.add(reps)
faiss.write_index(index, out_faiss)
print(f"Saved FAISS index → {out_faiss}")

with open(out_lookup, "w") as f:
    json.dump(lookup, f)
print(f"Saved lookup → {out_lookup}")