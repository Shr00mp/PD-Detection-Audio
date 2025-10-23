import pandas as pd
from feature_extraction import getFormants
from feature_extraction import getMFCCs
from feature_extraction import getAllFeatures
import os

data = pd.read_csv("acoustic-feature-models/audio_files.csv")

# === 2. Create a list to store feature dictionaries ===
features_list = []

# === 3. Loop through each audio file ===
for idx, row in data.iterrows():
    audio_ID = row["Sample ID"]
    # example path: audio_files\AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav
    audio_path = "audio_files/" + audio_ID + ".wav"
    feature_dict = getAllFeatures(audio_path, 50, 500, "Hertz")#
    feature_dict["Sample ID"] = audio_ID
    features_list.append(feature_dict)
