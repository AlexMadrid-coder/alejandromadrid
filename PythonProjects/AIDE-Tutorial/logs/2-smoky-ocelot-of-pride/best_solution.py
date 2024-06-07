import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Preprocessing
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_data[features]
y = train_data["Survived"]
X_test = test_data[features]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy="median")

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, ["Age", "Fare"]),
        (
            "cat",
            categorical_transformer,
            ["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
        ),
    ]
)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Split data into train and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Preprocessing of training data, fit model
pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = pipeline.predict(X_valid)

# Evaluate the model
score = accuracy_score(y_valid, preds)
print("Validation accuracy:", score)

# Preprocessing of test data, fit model
preds_test = pipeline.predict(X_test)

# Save test predictions to file
output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": preds_test})
output.to_csv("./working/submission.csv", index=False)
