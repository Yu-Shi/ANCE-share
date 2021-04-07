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


class DocumentDataset(IterableDataset):
    def __init__(self, documents, tokenizer, max_seq_length) -> None:
        super(DocumentDataset).__init__()
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__(self):
        def _pad_input_ids_with_mask(input_ids, max_length,
                  pad_token=0):
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
        
        for doc in documents:
            token_ids = tokenizer.encode(doc["text"], add_special_tokens=True, max_length=self.max_seq_length)
            token_ids, token_id_mask = _pad_input_ids_with_mask(token_ids, self.max_seq_length)
            yield {"docid": doc["id"], "input_ids": torch.tensor(token_ids, dtype=torch.long), "attention_mask": torch.tensor(token_id_mask, dtype=torch.long)}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--per_gpu_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--out_index_dir")
    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_batch_size = args.n_gpu * args.per_gpu_batch_size
    _, tokenizer, model = load_model(args, args.model_path)
    model.eval()

    logger.info("n_gpu = %d", args.n_gpu)
    logger.info("Total batch size = %d", total_batch_size)

    documents = []
    with open(args.collection, "r") as f:
        for line in f:
            documents.append(json.loads(line))

    dataset = DocumentDataset(documents, tokenizer, args.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=args.per_gpu_batch_size, num_workers=0)
    doc_ids = []
    embeddings = np.memmap(os.path.join(args.out_index_dir, "embeddings"), dtype='float32', mode='w+', shape=(len(documents), 768))
    idx = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            # print(data["input_ids"].shape)
            doc_id = data["docid"]
            doc_ids.extend(doc_id)
            batch_size = len(doc_id)
            input_ids, input_id_mask = data["input_ids"].to(args.device), data["attention_mask"].to(args.device)
            embs = model(input_ids, input_id_mask)
            embs = embs.detach().cpu().numpy()
            # print(embs.shape)
            embeddings[idx:idx+batch_size, :] = embs
            idx += batch_size

    assert idx == len(documents) == embeddings.shape[0]
    embeddings.flush()
    with open(os.path.join(args.out_index_dir, "docid"), "w") as f:
        for did in doc_ids:
            f.write(did + "\n")

    # build index
    cpu_index = faiss.IndexFlatIP(768)
    cpu_index.add(embeddings)
    faiss.write_index(cpu_index, os.path.join(args.out_index_dir, "index"))
