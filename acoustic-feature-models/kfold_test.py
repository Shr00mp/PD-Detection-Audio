import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("acoustic-feature-models/audio_features.csv")
X = df.drop(columns=["Sample ID", "Label"]) # features
Y = df["Label"] # labels
X.to_numpy()
Y.to_numpy()
print(f"X shape: {X.shape}")
print(f"Y.shape: {Y.shape}")

# Creates KFold object that divides dataset indices into n=5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}")
    print(f"  Train: index={train_index}, shape={train_index.shape}")
    print(f"  Test: index={test_index}, shape={test_index.shape}")