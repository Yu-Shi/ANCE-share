# ANCE-share

This repo contains code for ANCE document index generation and searching.

The document collection should be in the following jsonl format:

```
{"id": "<document id>", "text": "<document text>"}
```

I provide a demo collection in `dataset`.

Download the ANCE checkpoint at: https://github.com/microsoft/ANCE. The code is tested with the `ANCE(FirstP)` checkpoint.

## Index Generation

```bash
python build_index.py --collection=<document collection path>  --model_path=<ANCE checkpoint> --out_index_dir=<where to place document index and embeddings>    
```

For example:

```bash
python build_index.py --collection=dataset/demo_collection.jsonl  --model_path=encoder/some_checkpoint --out_index_dir=index_dir    
```

This step generates three files in `out_index_dir`, `docid`, `embeddings` and `index`. `embeddings` is a numpy array, and each row of it is the representation of a document, whose original id can be found at the same row in `docid`. `index` is the faiss index.

## Searching

```bash
python search.py --model_path=<ANCE checkpoint>  --index_dir=<your index dir>  --collection=<document collection path> --topk=<how many docs to be retrieved>  --query=<your query>  --use_gpu
```

If you set `use_gpu`, then the nearest neighbor search will be performed on GPUs. In this code, I use the first gpu for query embedding and other gpus for searching. For example, if you run the following code, 

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python search.py --model_path=encoder/some_checkpoint  --index_dir=index_dir  --collection=dataset/demo_collection.jsonl --topk=10  --use_gpu
```

GPU 0 will perform query embedding and GPU 1, 2 and 3 will perform searching. GPU brings significant acceleration, so it is recommended to set this flag on. Please note that performing search on GPUs consumes lots of GPU resources, and you may need to estimate how many GPUs will be used according to the index size and your GPU memory before running the code.