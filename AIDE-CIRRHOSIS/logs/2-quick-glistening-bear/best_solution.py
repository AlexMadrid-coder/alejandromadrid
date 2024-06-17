import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("./input/liver_cirrhosis.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = data.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data into features and target
X = data.drop("Stage", axis=1)
y = data["Stage"]

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Parameter grid for RandomForest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

# Initialize and train the Random Forest classifier with GridSearchCV
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_classifier, param_grid=param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

# Best estimator
best_rf = grid_search.best_estimator_

# Predict on validation set
y_val_pred = best_rf.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_val_pred)
f1_scores = classification_report(y_val, y_val_pred, output_dict=True)["weighted avg"][
    "f1-score"
]

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation F1-Score: {f1_scores:.4f}")

# Prepare submission of test predictions
y_test_pred = best_rf.predict(X_test)
submission = pd.DataFrame({"Stage": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
