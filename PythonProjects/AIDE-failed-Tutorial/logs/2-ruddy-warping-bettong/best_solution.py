import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_log_error, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the datasets
df1 = pd.read_csv("./input/diabetes_012_health_indicators_BRFSS2015.csv")
df2 = pd.read_csv("./input/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# Prepare the first dataset
X = df1.drop("Diabetes_012", axis=1)
y = df1["Diabetes_012"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the evaluation metrics
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
f1 = f1_score(y_test, y_pred, average="macro")
accuracy = accuracy_score(y_test, y_pred)

print(f"RMSLE: {rmsle}, F1-Score: {f1}, Accuracy: {accuracy}")

# Apply the model to the second dataset
X_new = df2.drop("Diabetes_binary", axis=1)
X_new_scaled = scaler.transform(X_new)
y_new_pred = model.predict(X_new_scaled)

# Save the predictions
submission = pd.DataFrame({"Diabetes_binary": y_new_pred})
submission.to_csv("./working/submission.csv", index=False)
