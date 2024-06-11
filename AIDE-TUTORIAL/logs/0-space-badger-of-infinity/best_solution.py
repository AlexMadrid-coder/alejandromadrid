import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_log_error, f1_score, accuracy_score
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("./input/water_potability.csv")

# Impute missing values
imputer = SimpleImputer(strategy="median")
data_imputed = imputer.fit_transform(data)
data = pd.DataFrame(data_imputed, columns=data.columns)

# Split data into features and target
X = data.drop("Potability", axis=1)
y = data["Potability"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize and train the SVM classifier
svm_model = SVC(kernel="rbf", gamma="auto")
svm_model.fit(X_train_scaled, y_train)

# Predict on validation set
y_pred = svm_model.predict(X_val_scaled)

# Calculate RMSLE, F1-score, and accuracy
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
f1 = f1_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)

# Print the evaluation metrics
print(f"RMSLE: {rmsle}, F1-score: {f1}, Accuracy: {accuracy}")
