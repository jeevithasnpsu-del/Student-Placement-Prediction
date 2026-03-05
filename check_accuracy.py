import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove StudentID
if "StudentID" in data.columns:
    data = data.drop("StudentID", axis=1)

# Convert Yes/No to 1/0 if exists
data = data.replace({
    "Yes": 1,
    "No": 0,
    "Placed": 1,
    "NotPlaced": 0
})

# Separate input and output
X = data.drop("PlacementStatus", axis=1)
y = data["PlacementStatus"]

# Convert everything to numeric
X = X.astype(float)
y = y.astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy =", round(accuracy * 100, 2), "%")