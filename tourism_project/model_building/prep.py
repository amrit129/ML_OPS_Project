import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from the repository data folder
df = pd.read_csv("tourism_project/data/tourism.csv")

# Remove unnecessary identifier/index columns
cols_to_drop = ["Unnamed: 0", "CustomerID"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Clean inconsistent categorical values (e.g., 'Fe Male' -> 'Female')
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Separate features (X) and target (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split into training and testing sets (80/20 stratified split)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally as CSV files matching the workflow artifact paths
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print(f"Xtrain shape: {Xtrain.shape}, Xtest shape: {Xtest.shape}")
