import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, f1_score, accuracy_score
import numpy as np

# Load the data
data = pd.read_csv("./input/water_potability.csv")

# Impute missing values
imputer = IterativeImputer(random_state=0)
data_imputed = imputer.fit_transform(data)
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Split data into features and target
X = data_imputed.drop("Potability", axis=1)
y = data_imputed["Potability"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Calculate RMSLE, F1-score, and accuracy
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred_proba))
f1 = f1_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)

# Print the evaluation metrics
print(f"RMSLE: {rmsle}")
print(f"F1-score: {f1}")
print(f"Accuracy: {accuracy}")
