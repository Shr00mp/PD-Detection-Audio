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
import matplotlib.pyplot as plt

class KNNModel:
    def __init__(self, X, Y, do_normalisation, do_feature_selection,
                 feature_selection_type, num_features):
        self = self
        self.X = X
        self.Y = Y
        self.do_normalisation = do_normalisation
        self.do_feature_selection = do_feature_selection
        self.feature_selection_type = feature_selection_type
        self.num_features = num_features
        self.best_k = None
        self.best_accuracy = None
    
    def normalise(self, X_df):
        # z-score normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)
        return X_scaled_df
    
    # filter method with f_classif test
    def univariate_selection(self, X_train, y_train, X_test, get_features):
        select_k_best = SelectKBest(score_func=f_classif, k=self.num_features)
        # fit to x_train + transform
        X_train_best = select_k_best.fit_transform(X_train, y_train)
        # DON'T fit to x_test, ONLY transform 
        X_test_best = select_k_best.transform(X_test)
        # print("Filter method selected features:\n" + "\n".join(X_train.columns[select_k_best.get_support()]))
        if (get_features): return X_train.columns[select_k_best.get_support()]
        else: return X_train_best, X_test_best

    # wrapper method with logistic regression
    def recursive_selection(self, X_train, y_train, X_test, get_features):
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        rfe = RFE(model, n_features_to_select=self.num_features)
        X_train_best = rfe.fit_transform(X_train, y_train)
        X_test_best = rfe.transform(X_test)
        # Note: rfe.get_support() is a boolean mask
        # X_train.columns[rfe.get_support()] has datatype pandas.Index 
        # print("Wrapper method selected features:\n" + "\n".join(X_train.columns[rfe.get_support()]))
        if (get_features): return X_train.columns[rfe.get_support()]
        else: return X_train_best, X_test_best
    
    # embedded method with random forest 
    def rf_selection(self, X_train, y_train, X_test, get_features):
        # Train rf and get feature importances
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        importances = model.feature_importances_

        # Print importances of features in decreasing order
        feature_importances = pd.Series(importances, index=X_train.columns)
        sorted_importances = feature_importances.sort_values(ascending=False)
        # print("\nImportance of features using rf: ")
        # print(sorted_importances)

        # Get and return x_train and x_test
        top_features = sorted_importances.head(self.num_features)
        top_feature_names = top_features.index
        X_train_best = X_train[top_feature_names]
        X_test_best = X_test[top_feature_names]
        if (get_features): return top_feature_names
        else: return X_train_best, X_test_best
    
    def intersection_selection(self, X_train, y_train, X_test):
        # Each variable below has data type pandas.Index
        univariate_features = self.univariate_selection(X_train, y_train, X_test, get_features=True)
        recursive_features = self.univariate_selection(X_train, y_train, X_test, get_features=True)
        rf_features = self.rf_selection(X_train, y_train, X_test, get_features=True)

        combined_indexes = univariate_features.union(recursive_features).union(rf_features)
        X_train_best = X_train[combined_indexes]
        X_test_best = X_test[combined_indexes]
        return X_train_best, X_test_best

    
    # model with k neighbours. returns mean accuracy
    def evaluate_model(self, k):
        X_df = self.X
        Y_df = self.Y
        # normalisation
        if (self.do_normalisation): X_df = self.normalise(X_df)
        Y_df.to_numpy()

        # Creates KFold object that divides dataset indices into n=5 folds
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []

        for i, [train_index, test_index] in enumerate(kf.split(X_df)):
            X_train = X_df.iloc[train_index]
            X_test = X_df.iloc[test_index]
            y_train = Y_df[train_index]
            y_test = Y_df[test_index]

            if (self.do_feature_selection) :
                if (self.feature_selection_type == "uni"):
                    X_train, X_test = self.univariate_selection(X_train, y_train, X_test, get_features=False)
                elif (self.feature_selection_type == "recursive"):
                    X_train, X_test = self.recursive_selection(X_train, y_train, X_test, get_features=False)
                elif (self.feature_selection_type == "rf"):
                    X_train, X_test = self.rf_selection(X_train, y_train, X_test, get_features=False)
                elif (self.feature_selection_type == "intersection"):
                    X_train, X_test = self.intersection_selection(X_train, y_train, X_test)
                else: 
                    print("Feature selection method does not exist. " \
                "\nChoose either uni, recursive, rf or intersection.")
                    return

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train) # fits model using training data
            y_pred = knn.predict(X_test) # makes predictions

            fold_accuracies.append(accuracy_score(y_test, y_pred))
            fold_precisions.append(precision_score(y_test, y_pred, pos_label="PwPD"))
            fold_recalls.append(recall_score(y_test, y_pred, pos_label="PwPD"))
            fold_f1_scores.append(f1_score(y_test, y_pred, pos_label="PwPD"))
        
        # print(f"\nFold #{i+1}:")
        # print(f"Accuracies: {fold_accuracies}")
        # print(f"Precisions: {fold_precisions}")
        # print(f"Recalls: {fold_recalls}")
        # print(f"F1 scores: {fold_f1_scores}")
        mean_accuracy = np.mean(fold_accuracies)
        # mean_prec = np.mean(fold_precisions)
        # mean_rec = np.mean(fold_recalls)
        # mean_f1 = np.mean(fold_f1_scores)
        # print(f"Average accuracy: {mean_accuracy}")
        # print(f"Average precision: {mean_prec}")
        # print(f"Average recall: {mean_rec}")
        # print(f"Average f1 score: {mean_f1}")

        return mean_accuracy
    
    def plot_k_accuracy(self, k_max):
        accuracies = []
        k_values = range(1, k_max)
        self.best_k = 1
        self.best_accuracy = 0
        for k in k_values:
            acc_score = self.evaluate_model(k)
            if (acc_score > self.best_accuracy): 
                self.best_accuracy = acc_score
                self.best_k = k
            accuracies.append(acc_score)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
        ax.set_title('KNN Accuracy vs. Number of Neighbors (k)')
        ax.set_xlabel('Number of Neighbors (k)')
        ax.set_ylabel('Accuracy')
        ax.grid(True)
        plt.close(fig)
        return fig


# df = pd.read_csv("acoustic-feature-models/audio_features.csv")
# X = df.drop(columns=["Sample ID", "Label"])  # Features
# Y = df["Label"]  # Labels

# knn_model = KNNModel(X, Y, True, True, "recursive", 13)
# knn_model.plot_k_accuracy(30)
# print(f"\nBest k: {knn_model.best_k} \nAccuracy: {knn_model.best_accuracy}")
