import torch
import pickle
import vocab
import os
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))
from spcl.model import CLModel, gen_all_reps, score_func
from spcl.config import *
from torch.utils.data import DataLoader
from spcl.data_process import load_iemocap_turn, build_dataset


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../resource/spcl_checkpoint")
MODEL_PATH = os.path.join(CHECKPOINT_PATH, "models/f1_0.6301_@epoch6.pkl")
# test_data_path = "/Users/elnuralimirzayev/Thesis/notebooks/eahris/resource/spcl_test/test_data.json"


def load_model_and_run(model_path, checkpoint_path, dialogues):
    if model_path.endswith(".pkl"):
        model = torch.load(
            model_path, map_location=torch.device("cpu"), pickle_module=pickle
        )
    else:
        model = torch.load(model_path, map_location=torch.device("cpu"))
    centers = torch.load(checkpoint_path + "/temp/centers.pkl", map_location=torch.device("cpu"), pickle_module=pickle)
    centers_mask = torch.load(checkpoint_path + "/temp/centers_mask.pkl", map_location=torch.device("cpu"), pickle_module=pickle)
    emo_list = []
    for batch_id, batch_data in enumerate(dialogues):
        sentences = batch_data[0]
        with torch.no_grad():
            ccl_reps = model.gen_f_reps(sentences)
        features = [ccl_reps]
        for idx, feature in enumerate(features):
            prediction = model(feature, centers, score_func)
            prediction -= (1 - centers_mask) * 2
            emo_list.append(torch.argmax(prediction.max(-1)[0], -1))
    return emo_list

def idx2emo(idx_list):
    ev_pkl = torch.load(CHECKPOINT_PATH + "/vocabs/emotion_vocab.pkl", map_location=torch.device("cpu"), pickle_module=pickle)
    emotion_vocab = vocab.Vocab(words=ev_pkl['index2word'])
    return [emotion_vocab.index2word(idx) for idx in idx_list]

def spcl_run(chat_history):
    dialogues = build_dataset(chat_history)
    emo_eval = load_model_and_run(MODEL_PATH, CHECKPOINT_PATH, dialogues)
    return idx2emo(emo_eval)

