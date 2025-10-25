import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# filter method using chi-square test
def univariate_selection(X_train, y_train, X_test):
    select_k_best = SelectKBest(score_func=f_classif, k=10)
    # fit to x_train + transform
    X_train_best = select_k_best.fit_transform(X_train, y_train)
    # DON'T fit to x_test, ONLY transform 
    X_test_best = select_k_best.transform(X_test)
    print(f"Selected features: {X_train.columns[select_k_best.get_support()]}")
    return X_train_best, X_test_best

# wrapper method with logistic regression
def recursive_selection(X_train, y_train, X_test):
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=10)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)
    print(f"Selected features: {X_train.columns[rfe.get_support()]}")
    return X_train_rfe, X_test

# embedded method with random forest 
def rf_selection(X_train, y_train):
    # Train rf and get feature importances
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    importances = model.feature_importances_

    # Print importances of features in decreasing order
    feature_importances = pd.Series(importances, index=X_train.columns)
    print(feature_importances.sort_values(ascending=False))

df = pd.read_csv("acoustic-feature-models/audio_features.csv")
X = df.drop(columns=["Sample ID", "Label"]) # features
Y = df["Label"] # labels


# z-score normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# training and testing split of 80-20 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Filter method
X_train_best, X_test_best = univariate_selection(X_train, y_train, X_test)

X = X.to_numpy()
Y = Y.to_numpy()
