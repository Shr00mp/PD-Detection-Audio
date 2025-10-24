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

# Creates KFold object that divides dataset indices into n=5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, [train_index, test_index] in enumerate(kfold.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc_score)

print(f"Accuracies: {fold_accuracies}")
mean_accuracy = np.mean(fold_accuracies)
print(f"Average accuracy: {mean_accuracy}")
