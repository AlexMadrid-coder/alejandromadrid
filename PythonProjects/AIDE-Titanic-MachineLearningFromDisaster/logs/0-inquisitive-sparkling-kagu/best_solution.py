import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data by selecting features and dealing with missing values
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_data[features]
y = train_data["Survived"]
X_test = test_data[features]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X["Age"] = imputer.fit_transform(X[["Age"]])
X_test["Age"] = imputer.transform(X_test[["Age"]])

imputer_fare = SimpleImputer(strategy="median")
X["Fare"] = imputer_fare.fit_transform(X[["Fare"]])
X_test["Fare"] = imputer_fare.transform(X_test[["Fare"]])

# Impute Embarked with the mode
mode_embarked = train_data["Embarked"].mode()[0]
X["Embarked"].fillna(mode_embarked, inplace=True)
X_test["Embarked"].fillna(mode_embarked, inplace=True)

# Encode categorical data
encoder = LabelEncoder()
X["Sex"] = encoder.fit_transform(X["Sex"])
X_test["Sex"] = encoder.transform(X_test["Sex"])

X["Embarked"] = encoder.fit_transform(X["Embarked"])
X_test["Embarked"] = encoder.transform(X_test["Embarked"])

# Splitting train data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with Grid Search
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# Validate the model
y_pred = grid_search.best_estimator_.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Predict on test set
predictions = grid_search.best_estimator_.predict(X_test)

# Create submission file
output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("./working/submission.csv", index=False)
