import torch
import torch.nn as nn
from transformers import AutoModel
from sklearn.metrics import euclidean_distances
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from spcl.config import CONFIG


def score_func(x, y):
    return (1+F.cosine_similarity(x, y, dim=-1))/2 + 1e-8


def gen_all_reps(model, data):
    model.eval()
    results = []
    label_results = []

    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=0,  # multiprocessing.cpu_count()
    )
    inner_model = model.module if hasattr(model, "module") else model
    tq_train = tqdm(total=len(dataloader), position=1)
    tq_train.set_description("generate representations for all data")
    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            batch_data = [x.to(inner_model.device()) for x in batch_data]
            sentences = batch_data[0]
            emotion_idxs = batch_data[1]

            outputs = inner_model.gen_f_reps(sentences)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            for idx, label in enumerate(emotion_idxs.reshape(-1)):
                if label < 0:
                    continue
                results.append(outputs[idx])
                label_results.append(label)
            tq_train.update()
    tq_train.close()
    dim = results[0].shape[-1]

    results = torch.stack(results, 0).reshape(-1, dim)
    label_results = torch.stack(label_results, 0).reshape(-1)

    return results, label_results


class CLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config["dropout"]
        self.num_classes = config["num_classes"]
        self.pad_value = config["pad_value"]
        self.mask_value = config["mask_value"]
        self.f_context_encoder = AutoModel.from_pretrained(
            config["bert_path"], local_files_only=False
        )
        num_embeddings, self.dim = (
            self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        )
        self.f_context_encoder.resize_token_embeddings(num_embeddings + 256)
        self.predictor = nn.Sequential(nn.Linear(self.dim, self.num_classes))
        self.g = nn.Sequential(
            nn.Linear(self.dim, self.dim),
        )

    def device(self):
        return self.f_context_encoder.device

    def gen_f_reps(self, sentences):
        """
        generate vector representations for each turn of conversation
        """
        batch_size, max_len = sentences.shape[0], sentences.shape[-1]
        sentences = sentences.reshape(-1, max_len)
        mask = 1 - (sentences == self.pad_value).long()
        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )["last_hidden_state"]
        mask_pos = (sentences == self.mask_value).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos, :]
        # feature = torch.dropout(mask_outputs, 0.1, train=self.training)
        feature = mask_outputs
        if self.config["output_mlp"]:
            feature = self.g(feature)
        return feature

    def forward(self, reps, centers, score_func):

        num_classes, num_centers = centers.shape[0], centers.shape[1]
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_centers, -1)
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_classes, num_centers, -1)

        centers = centers.unsqueeze(0).expand(reps.shape[0], -1, -1, -1)
        # batch * turn, num_classes, num_centers
        sim_matrix = score_func(reps, centers)

        # batch * turn, num_calsses
        scores = sim_matrix
        return scores
