import os
import pandas as pd
import csv
from alive_progress import alive_bar
from speechbrain.inference.interfaces import foreign_class
import matplotlib.pyplot as plt

classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
)

def run_speechbrain(report_path: str, eval_report_path: str):
    # Read the CSV file from report_path
    with open(report_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="|")
        hist = {"ang": 0, "hap": 0, "sad": 0, "neu": 0}
        missclassified = {"ang": 0, "hap": 0, "sad": 0, "neu": 0}
        recognized = []
        total = 0
        correct = 0
        for row in reader:
            total += 1
            out_prob, score, index, text_lab = classifier.classify_file(
                row["output_wav"]
            )
            recognized.append(text_lab[0])
            hist[text_lab[0]] += 1
            if row["initial_emo"] == text_lab[0]:
                correct += 1
            else:
                missclassified[row["initial_emo"]] += 1
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.2f}")

    # Generate and save the report
    plt.figure(figsize=(12, 8))

    # Plot histogram
    plt.subplot(2, 2, 1)
    plt.bar(hist.keys(), hist.values(), color='blue')
    plt.title("Emotion Histogram")
    plt.xlabel("Emotion")
    plt.ylabel("Count")

    # Plot misclassified counts
    plt.subplot(2, 2, 2)
    plt.bar(missclassified.keys(), missclassified.values(), color='red')
    plt.title("Misclassified Counts")
    plt.xlabel("Emotion")
    plt.ylabel("Count")

    # Plot accuracy
    plt.subplot(2, 2, 3)
    plt.bar(["Accuracy"], [accuracy], color='green')
    plt.title("Accuracy")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")

    # Save the report to a file
    plt.tight_layout()
    plt.savefig(eval_report_path)
    plt.close()

run_speechbrain("/Users/elnuralimirzayev/Thesis/notebooks/eahris/tts_output/feed.csv", 
                "/Users/elnuralimirzayev/Thesis/notebooks/eahris/report.png")
