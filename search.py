import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import argparse
import json
import logging
import numpy as np
import os
from tqdm import tqdm
import faiss
from model.models import MODEL_CLASSES

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

ngpu = faiss.get_num_gpus()

gpu_resources = []
tempmem = -1

for i in range(ngpu):
    res = faiss.StandardGpuResources()
    if tempmem >= 0:
        res.setTempMemory(tempmem)
    gpu_resources.append(res)

def make_vres_vdev(i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    # args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES["rdot_nll"]
    config = config_class.from_pretrained(
        checkpoint_path,
        num_labels=num_labels,
        finetuning_task="MSMarco",
        cache_dir=None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        checkpoint_path,
        do_lower_case=True,
        cache_dir=None,
    )
    model = model_class.from_pretrained(
        checkpoint_path,
        from_tf=bool(".ckpt" in checkpoint_path),
        config=config,
        cache_dir=None,
    )
    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return config, tokenizer, model


def load_index(index_dir: str):
    def _load_docids(docid_path: str):
        docids = [line.rstrip() for line in open(docid_path, 'r').readlines()]
        return docids
    index_path = os.path.join(index_dir, 'index')
    docid_path = os.path.join(index_dir, 'docid')
    index = faiss.read_index(index_path)
    docids = np.array(_load_docids(docid_path))
    return index, docids


def search(query, max_query_length, device, topk, use_gpu, id_to_document, tokenizer, model, index, docids):
    def pad_input_ids_with_mask(input_ids, max_length, pad_token=0):
        padding_length = max_length - len(input_ids)
        padding_id = [pad_token] * padding_length

        attention_mask = []

        if padding_length <= 0:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length

        return input_ids, attention_mask

    def get_docid_and_text(candidate_list, docids, id_to_document):
        candidate_ids = docids[candidate_list]
        docs = []
        for _id in candidate_ids:
            docs.append(id_to_document[_id])
        
        return candidate_ids, docs

    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True 
        gpu_vector_resources, gpu_devices_vector = make_vres_vdev(1, ngpu)
        index = faiss.index_cpu_to_gpu_multiple(gpu_vector_resources, gpu_devices_vector, index, co)

    model.eval()
    token_ids = tokenizer.encode(query, add_special_tokens=True, max_length=max_query_length)
    token_ids, token_id_mask = pad_input_ids_with_mask(token_ids, max_query_length)
    token_ids, token_id_mask = torch.unsqueeze(torch.tensor(token_ids, dtype=torch.long), 0).to(device), torch.unsqueeze(torch.tensor(token_id_mask, dtype=torch.long), 0).to(device)
    with torch.no_grad():
        emb = model(token_ids, token_id_mask)
        emb = emb.detach().cpu().numpy()
    D, I = index.search(emb, topk)
    D, I = np.squeeze(D), np.squeeze(I)
    candidate_ids, docs = get_docid_and_text(I, docids, id_to_document)
    return candidate_ids, docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--index_dir")
    parser.add_argument("--query", type=str, default="The line between Mariahütte and Wadern")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    logger.info("Loading document collection...")
    id_to_document = {}
    with open(args.collection, "r") as f:
        for line in f:
            obj = json.loads(line)
            id_to_document[obj["id"]] = obj["text"]

    args.n_gpu = 1  # for pytorch
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tokenizer, model = load_model(args, args.model_path)
    index, docids = load_index(args.index_dir)

    candidate_ids, docs = search(args.query, args.max_query_length, args.device, args.topk, args.use_gpu, id_to_document, tokenizer, model, index, docids)
    for did, doc in zip(candidate_ids, docs):
        print(int(did), doc)