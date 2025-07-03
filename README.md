# Logistic-Regressioin-assignment

### 1. Data Collection
Gather labeled data (input features and binary output: 0 or 1).

Example: Predict if an email is spam (1) or not spam (0).

 ### 2. Data Preprocessing
Handle Missing Values: Fill or drop missing data.

Encoding: Convert categorical variables to numeric (e.g., using One-Hot Encoding).

Feature Scaling: Normalize/standardize values if needed.

 ### 3. Exploratory Data Analysis (EDA)
Understand relationships using graphs like histograms, pair plots, and correlation matrices.

Detect outliers or imbalances in the data.

### 4. Feature Selection
Remove irrelevant or redundant features.

Use techniques like correlation checks, mutual information, or model-based selection.

### 5. Model Training
Split data into training and testing sets.

Fit the logistic regression model on training data.

The model estimates parameters (β0, β1, ..., βn) using Maximum Likelihood Estimation (MLE) to minimize the loss function (typically log loss).

###  6. Model Prediction
Predict probabilities using the sigmoid function

​Classify outputs using a threshold (e.g., 0.5 → class 1, otherwise 0).

### 7. Model Evaluation
Use metrics:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

ROC Curve & AUC

 ### 8. Hyperparameter Tuning (Optional)
Adjust regularization parameter (C) or penalty type (l1 or l2) to improve performance.

Use GridSearchCV or RandomSearchCV.

### 9. Model Deployment (Optional)
Save and export the model (using pickle or joblib).

Deploy it in a web app or API for predictions.
