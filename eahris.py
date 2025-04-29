import configparser
import json
import os
import csv
import shutil
import time
from alive_progress import alive_bar
from llm.openai import OpenaiApi
from spcl.spcl import spcl_run
from eatts import tts_client
from eval.run_speechbrain import run_speechbrain

emolabel_map = {
    "ang": "[angry] ",
    "hap": "[happy] ",
    "exc": "[happy] ",
    "sad": "[sad] ",
    "fru": "[sad] ",
    "neu": "", # should be empty
}

emo_map = {
    "ang": "ang",
    "hap": "hap",
    "exc": "hap",
    "sad": "sad",
    "fru": "sad",
    "neu": "neu"
}

def read_json_file(path):
    """
    Reads a JSON file from the specified path. The JSON file is expected to contain
    an array of arrays of objects with "text" and "speaker" fields. Prints the fields
    of every object.

    Args:
        path (str): The path to the JSON file.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        dialogues = []
        for array in data:
            dialogue = []
            for obj in array:
                if obj.get("label") == "fru":
                    obj["label"] = "sad"
                elif obj.get("label") == "exc":
                    obj["label"] = "hap"
                dialogue.append(obj)
            dialogues.append(dialogue)
        return dialogues

def load_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

def initialize_modules(config, is_benchmark_baseline=False):
    if not is_benchmark_baseline:
        apikey = config["secrets"]["OpenAiApiKey"]
        llm_client = OpenaiApi(api_key=apikey)
    else:
        llm_client = None
    spcl_client = spcl_run
    eatts_client = tts_client
    return llm_client, spcl_client, eatts_client

def add_to_chathistory(new_msg, user, chat_history, spcl_client):
    chat_history.append(
        {
            "speaker": user,
            "text": new_msg
        }
    )
    return spcl_client([chat_history])

def evaluate_output(tts_client, report_path):
    # Evaluate the output
    def callback(ch, method, properties, body):
        print("Received finish signal from TTS.")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_speechbrain(report_path, os.path.join(os.path.dirname(__file__), f"report_{timestamp}.png",))
    # Wait for TTS finish signal to start evaluation
    tts_client.wait_for_finish(callback)

def benchmark_baseline(spcl_client, eatts_client):
    # Remove contents of tts_output directory
    tts_output_dir = os.path.join(os.path.dirname(__file__), "tts_output")
    if os.path.exists(tts_output_dir):
        shutil.rmtree(tts_output_dir)
    os.makedirs(tts_output_dir, exist_ok=True)
    
    tts_feed = []
    dialogues = read_json_file("resource/spcl_test/test_data.json")
    with alive_bar(len(dialogues), title="Processing dialogues") as bar:
        for dialogue in dialogues:
            bar()
            emo_list = spcl_client([dialogue])
            for i, emo in enumerate(emo_list):
                if "label" in dialogue[i]:
                    tts_feed.append({
                        "text": emolabel_map[emo] + dialogue[i]["text"],
                        "emo": emo_map[emo],
                        "initial_emo": emo_map[dialogue[i]["label"]],
                        "initial_text": dialogue[i]["text"],
                        "output_wav": os.path.join(tts_output_dir, f"{dialogue[i]['speaker']}_{i}.wav")
                    })
    eatts_client.call_tts(tts_feed)
    # Export tts_feed to feed.csv
    output_csv_path = os.path.join(tts_output_dir, "feed.csv")
    with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["text", "emo", "initial_emo", "initial_text", "output_wav"], delimiter="|")
        writer.writeheader()
        writer.writerows(tts_feed)
    evaluate_output(eatts_client, output_csv_path)
    

def user_baseline(spcl_client, eatts_client, llm_client):
    chat_history = []
    emo_history = []
    print("Welcome to the Emotion-Aware Human-Robot Interaction System (EAHRIS)!")
    while True:
        user_input = input()
        emo_history = add_to_chathistory(user_input, "User", chat_history, spcl_client)
        llm_response = llm_client.chat(user_input, chat_history)
        emo_history = add_to_chathistory(llm_response.output_text, "NICO", chat_history, spcl_client)
        print(llm_response.output_text)
        eatts_client.call_tts(llm_response.output_text, emo_history[-1])

def main():
    config = load_config()
    is_benchmark_baseline = config.getboolean("baseline", "IsBenchmarkBaseline", fallback=False)
    llm_client, spcl_client, eatts_client = initialize_modules(config, is_benchmark_baseline)
    if is_benchmark_baseline:
        benchmark_baseline(spcl_client, eatts_client)
    else:
        user_baseline(spcl_client, eatts_client, llm_client)

if __name__ == "__main__":
    main()
