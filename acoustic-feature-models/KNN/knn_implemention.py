import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("acoustic-feature-models/audio_features.csv")
X = df.drop(columns=["Sample ID", "Label"]) # features
Y = df["Label"] # labels
X = X.to_numpy()
Y = Y.to_numpy()

# Creates KFold object that divides dataset indices into n=5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []

for i, [train_index, test_index] in enumerate(kf.split(X)):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = Y[train_index]
    y_test = Y[test_index]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    fold_accuracies.append(accuracy_score(y_test, y_pred))
    fold_precisions.append(precision_score(y_test, y_pred, pos_label="PwPD"))
    fold_recalls.append(recall_score(y_test, y_pred, pos_label="PwPD"))
    fold_f1_scores.append(f1_score(y_test, y_pred, pos_label="PwPD"))
    

print(f"Accuracies: {fold_accuracies}")
print(f"Precisions: {fold_precisions}")
print(f"Recalls: {fold_recalls}")
print(f"F1 scores: {fold_f1_scores}")
mean_accuracy = np.mean(fold_accuracies)
mean_prec = np.mean(fold_precisions)
mean_rec = np.mean(fold_recalls)
mean_f1 = np.mean(fold_f1_scores)
print(f"Average accuracy: {mean_accuracy}")
print(f"Average precision: {mean_prec}")
print(f"Average recall: {mean_rec}")
print(f"Average f1 score: {mean_f1}")
