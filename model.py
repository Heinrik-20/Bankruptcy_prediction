import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

file_dir = r"../input/company-bankruptcy-prediction/data.csv"
dataset = pd.read_csv(file_dir)

labelled_features = [" Net Income Flag", " Liability-Assets Flag"]
dataset = dataset.drop(labels=labelled_features, axis=1)
dataset.describe()

X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
# Scale features to the same range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
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
log_reg = LogisticRegression()
lreg_model = log_reg.fit(X_train, y_train)
lreg_pred = lreg_model.predict(X_test)
print(classification_report(y_test, lreg_pred))


# Create oversampling dataset using SMOTE
labels_og = dataset["Bankrupt?"]
X_og = dataset.drop(["Bankrupt?"], axis=1)
sampler = SMOTE()
X_smote, y_smote = sampler.fit_resample(X_og, labels_og)

# Scale data
scaler = StandardScaler()
X_smote_scaled = scaler.fit_transform(X_smote)
X_smote_scaled = pd.DataFrame(X_smote_scaled, columns=X_smote.columns)
# Splitting data
X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_smote_scaled, y_smote, 
                                                                            stratify=y_smote, test_size=0.3, 
                                                                            shuffle=True)

# Logistic Regression with Oversampled data
log_reg_over = LogisticRegression()
lreg_over_model = log_reg_over.fit(X_smote_train, y_smote_train)
lreg_over_pred = lreg_over_model.predict(X_smote_test)
print(classification_report(y_smote_test, lreg_over_pred))