import pandas as pd
from feature_extraction import getFormants
from feature_extraction import getMFCCs
from feature_extraction import getAllFeatures
import os

# Run only once

data = pd.read_csv("acoustic-feature-models/audio_files.csv")

# List to store feature dictionaries (for the separate audio files)
features_list = []

# Looping through audio files in data
for idx, row in data.iterrows():
    audio_ID = row["Sample ID"]
    # example path: audio_files\AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav
    audio_path = "audio_files/" + audio_ID + ".wav"
    feature_dict = getAllFeatures(audio_path, 50, 500, "Hertz")#
    feature_dict["Sample ID"] = audio_ID
    feature_dict["Label"] = row["Label"]
    features_list.append(feature_dict)

features_df = pd.DataFrame(features_list)
features_df.to_csv("acoustic-feature-models/audio_features.csv", index=False)
# Note: new csv file contains the Sample ID, which should not be fed into ML models