import argparse
import faiss
import h5py
import numpy as np
import os
from pathlib import Path
import time 
import datetime

def store_results(dst, algo, kind, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

def load_database(file):
    print("loading clip768 (converting Float16 -> Float32)")
    with h5py.File(file, 'r') as f:
        X = np.array(f['emb'], dtype=np.float32)
    # for col in X.T:
    #     normalize(col[:, np.newaxis], copy=False)
    return X

def run(kind, db_path, q_path, outdir, size="300K", k=30):
    print("Running", kind)
    
    data = load_database(db_path)
    queries = load_database(q_path)
    n, d = data.shape

    nlist = 128 # number of clusters/centroids to build the IVF from

    if kind.startswith("pca"):
        index_identifier = f"IVF{nlist},FlatIP"
        index = faiss.index_factory(d, index_identifier)
    elif kind.startswith("hamming"):
        index_identifier = f"BIVF{nlist},Flat" # use binary IVF index
        d = 64 * d # one chunk contains 64 bits
        index = faiss.index_binary_factory(d, index_identifier)
        # create view to interpret original uint64 as 8 chunks of uint8
        data = np.array(data).view(dtype="uint8")
        queries = np.array(queries).view(dtype="uint8")
    else:
        # if kind == "clip768":
        # convert vectors from float16 to float32 
        # normalize vectors
        # dot product / angle as distance (1-cosine) 
        raise Exception(f"unsupported input type {kind}")

    print(f"Training index on {data.shape}")
    start = time.time()
    index.train(data)
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    assert index.is_trained

    for nprobe in [1, 2, 5, 10, 20, 50, 100]:
        print(f"Starting search on {queries.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        I = I + 1 # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

        store_results(os.path.join(outdir, f"{identifier}.h5"), "faissIVF", kind, D, I, elapsed_build, elapsed_search, identifier, size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default="300K"
    )
    parser.add_argument(
        "--k",
        default=30,
    )

    args = parser.parse_args()

    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    db_path = f"data2024/laion2B-en-clip768v2-n={args.size}.h5"
    query_path = "data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
    output_dir = f"results/task1/{args.size}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run("pca32v2", db_path, query_path, output_dir, args.size, args.k)