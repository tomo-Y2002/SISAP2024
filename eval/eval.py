import h5py
import numpy as np
import os
import csv
from urllib.request import urlretrieve


def get_groundtruth(size="300K"):
    path = f"data2024/gold-standard-dbsize={size}--public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
    gt_f = h5py.File(path, "r")
    true_I = np.array(gt_f['knns'])
    gt_f.close()
    return true_I

def get_all_results(dirname):
    for root, _, files in os.walk(dirname):
        for fn in files:
            if os.path.splitext(fn)[-1] != ".h5":
                continue
            try:
                f = h5py.File(os.path.join(root, fn), "r")
                yield f
                f.close()
            except:
                print("Unable to read", fn)

def get_recall(I, gt, k):
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)

def return_h5_str(f, param):
    if param not in f:
        return 0
    x = f[param][()]
    if type(x) == np.bytes_:
        return x.decode()
    return x


if __name__ == "__main__":
    true_I_cache = {}

    columns = ["data", "size", "algo", "buildtime", "querytime", "params", "recall"]
    
    with open('res.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for res in get_all_results("results"):
            try:
                size = res.attrs["size"]
                d = dict(res.attrs)
            except: 
                size = res["size"][()].decode()
                d = {k: return_h5_str(res, k) for k in columns}
            if size not in true_I_cache:
                true_I_cache[size] = get_groundtruth(size)
            recall = get_recall(np.array(res["knns"]), true_I_cache[size], 10) # なぜか10 recall@10
            d['recall'] = recall
            print(d["data"], d["algo"], d["params"], "=>", recall)
            writer.writerow(d)