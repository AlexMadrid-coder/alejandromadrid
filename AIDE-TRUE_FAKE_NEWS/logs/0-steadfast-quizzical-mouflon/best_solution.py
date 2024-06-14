import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_log_error, confusion_matrix, roc_curve, auc
import numpy as np

# Load data
fake_news = pd.read_csv("./input/Fake.csv")
true_news = pd.read_csv("./input/True.csv")

# Label the data
fake_news["label"] = 0
true_news["label"] = 1

# Combine the datasets
data = pd.concat([fake_news, true_news], ignore_index=True)

# Select features and labels
X = data["text"]
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize the text data
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test_tfidf)
y_pred_proba = rf_classifier.predict_proba(X_test_tfidf)[:, 1]

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred_proba))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate ROC AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Print metrics
print("RMSLE:", rmsle)
print("Confusion Matrix:\n", conf_matrix)
print("ROC AUC:", roc_auc)

# Save predictions for evaluation
predictions = pd.DataFrame({"True_Label": y_test, "Predicted_Label": y_pred})
predictions.to_csv("./working/submission.csv", index=False)
