import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest

file_dir = r"../input/company-bankruptcy-prediction/data.csv"
dataset = pd.read_csv(file_dir)

labelled_features = [" Net Income Flag", " Liability-Assets Flag"]
dataset = dataset.drop(labels=labelled_features, axis=1)
dataset.describe()

def scale_data(scaler, X):
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled

def logistic_reg(X_train, X_test, y_train):
    log_reg = LogisticRegression()
    lreg_model = log_reg.fit(X_train, y_train)
    return lreg_model.predict(X_test)

X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
# Scale features to the same range
X_scaled = scale_data(StandardScaler(), X)
y = pd.DataFrame(y, columns=["Bankrupt?"])
new_dataset = y.join(X_scaled)

# Using the undersampled dataset
unstable = new_dataset.loc[new_dataset["Bankrupt?"] == 1]
stable = new_dataset.loc[new_dataset["Bankrupt?"] == 0][0:len(unstable)]
undersampled = pd.concat([unstable, stable])
labels = undersampled["Bankrupt?"]
undersampled = undersampled.drop(["Bankrupt?"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(undersampled, labels, stratify=labels, test_size=0.3, 
                                                   shuffle=True)

# Basic Logistic Regression with undersampled data
lreg_pred = logistic_reg(X_train, X_test, y_train)
print(classification_report(y_test, lreg_pred))

# Create oversampling dataset using SMOTE
labels_og = dataset["Bankrupt?"]
X_og = dataset.drop(["Bankrupt?"], axis=1)
sampler = SMOTE()
X_smote, y_smote = sampler.fit_resample(X_og, labels_og)

# Scale data
X_smote_scaled = scale_data(StandardScaler(), X_smote)
# Splitting data
X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_smote_scaled, y_smote, 
                                                                            stratify=y_smote, test_size=0.3, 
                                                                            shuffle=True)

# Logistic Regression with Oversampled data
lreg_over_pred = logistic_reg(X_smote_train, X_smote_test, y_smote_train)
print(classification_report(y_smote_test, lreg_over_pred))

# Try using lesser features through select_k_best
selector = SelectKBest(k=1)
X_new = selector.fit_transform(X_smote_scaled, y_smote)
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_smote, stratify=y_smote, test_size=0.3
                                                                   , shuffle=True)

lreg_kbest_pred = logistic_reg(X_new_train, X_new_test, y_new_train)
print(classification_report(y_new_test, lreg_kbest_pred))
