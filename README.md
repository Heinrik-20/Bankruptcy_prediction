A repository of bankruptcy prediction using Logistic Regression. Data was sourced from Kaggle.com.

The code starts with scaling the features with StandardScaler to prevent one feature from overpowering the other features. 
Next, only 220 examples from the stable dataset (non-bankrupt) was taken since there was only 220 bankrupt examples in the original dataset.
SMOTE was included to help generate more bankrupt examples since the dataset is biased towards examples where bankruptcy did not occured. 
Logistic Regression is then applied to both undersampled and oversampled datasets. 
