import torch 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("acoustic-feature-models/audio_features.csv")
X = df.drop(columns=["Sample ID", "Label"]) # features
Y = df["Label"] # labels
