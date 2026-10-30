import os
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

print("Dataset registered and validated successfully!")
print(f"Dataset shape: {df.shape}")
print("\nTarget Distribution (ProdTaken):")
print(df["ProdTaken"].value_counts())
print("\nDataset Summary:")
print(df.describe().T[["count", "mean", "std", "min", "max"]])
