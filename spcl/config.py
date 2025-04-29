from transformers import AutoTokenizer

CONFIG = {
    "bert_path": "princeton-nlp/sup-simcse-roberta-base",
    "num_classes": 6,
    "emotion_vocab": "/Users/elnuralimirzayev/Thesis/notebooks/eahris/resource/spcl_checkpoint/vocabs/emotion_vocab.pkl",
    "local_rank": 0,
    "max_len": 256,
    "pad_value": 1,
    "mask_value": 2,
    "dropout": 0.1,
    "batch_size": 128,
}

tokenizer = AutoTokenizer.from_pretrained(CONFIG["bert_path"], local_files_only=False)
_special_tokens_ids = tokenizer("<mask>")["input_ids"]
CLS = _special_tokens_ids[0]
CONFIG["CLS"] = CLS
