# Step 1: Import Libraries
import pandas as pd
import numpy as np

# Step 2: Load Dataset
df = pd.read_csv("Data.csv")

print("Original Dataset:\n")
print(df)

# Step 3: Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Step 4: Handle Missing Values

# Fill Age with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill Salary with mean
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

print("\nAfter Handling Missing Values:\n")
print(df)

# Step 5: Encode Categorical Data

# Convert Country into dummy variables
df = pd.get_dummies(df, columns=['Country'])

# Convert Purchased column (Yes=1, No=0)
df['Purchased'] = df['Purchased'].map({'Yes': 1, 'No': 0})

print("\nAfter Encoding:\n")
print(df)

# Step 6: Feature Scaling (Normalization)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print("\nAfter Scaling:\n")
print(df)

# Step 7: Save Cleaned Dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned dataset saved as cleaned_data.csv")