# SISAP2024
SISAP2024 development branch


## Steps for running

the following commands should be run

```bash
conda create -n faiss python=3.12
conda activate faiss
conda install -c pytorch faiss-cpu=1.8.0
conda install matplolib
conda install h5py
```

```bash
mkdir data2024
cd data2024
curl -O https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=$DBSIZE.h5
curl -O http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5  # this url will be updated soon
curl -O http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=$DBSIZE--public-queries-2024-laion2B-en-clip768v2-n=10k.h5 # this url will be updated soon
```

to demonstrate
```bash
python search/search.py
python eval/eval.py
python eval/plot.py res.csv
```