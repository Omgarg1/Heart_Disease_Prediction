import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ============================
# Load Dataset
# ============================
df = pd.read_csv('Heart_Disease_Dataset.csv')
target_col = 'Heart Disease Status'  # change if different

# ============================
# Encode Categorical Columns (store mappings)
# ============================
label_encoders = {}
category_mappings = {}

for col in df.columns:
    if col != target_col and not pd.api.types.is_numeric_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        category_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Fill missing values
num_cols = df.drop(target_col, axis=1).select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

non_num_cols = df.drop(target_col, axis=1).select_dtypes(exclude=['number']).columns
for col in non_num_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Define X and y
X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode target if needed
if not pd.api.types.is_numeric_dtype(y):
    y = LabelEncoder().fit_transform(y)

# ============================
# Split Data
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# Build Pipeline
# ============================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully with Accuracy: {accuracy:.4f}")

# ============================
# Save Model & Encoders
# ============================
pickle.dump({'model': pipeline, 'encoders': label_encoders}, open('heart_disease_model.pkl', 'wb'))
joblib.dump({'model': pipeline, 'encoders': label_encoders}, 'heart_disease_model.joblib')
print("💾 Model and encoders saved successfully.\n")

# ============================
# Interactive User Input
# ============================
print("=======================================")
print("💬 Heart Disease Risk Prediction System")
print("=======================================\n")

user_data = {}

for feature in X.columns:
    if feature in label_encoders:
        print(f"\n👉 {feature}")
        print("USE CODES GIVEN BELOW:- ")
        for category, code in category_mappings[feature].items():
            print(f"   {category} = {code}")
        while True:
            try:
                value = int(input(f"Enter {feature} (use above numeric code): "))
                if value in category_mappings[feature].values():
                    user_data[feature] = value
                    break
                else:
                    print("⚠️ Invalid code. Please choose from the listed options.")
            except ValueError:
                print("⚠️ Please enter a valid integer.")
    else:
        while True:
            try:
                value = float(input(f"Enter your {feature}: "))
                user_data[feature] = value
                break
            except ValueError:
                print("⚠️ Please enter a valid numeric value.")

# Convert user input to DataFrame
user_df = pd.DataFrame([user_data])

# ============================
# Make Prediction
# ============================
prediction = pipeline.predict(user_df)[0]
probability = pipeline.predict_proba(user_df)[0][1] if hasattr(pipeline, 'predict_proba') else None

# ============================
# Display Result
# ============================
print("\n=======================================")
if prediction == 1:
    print("🔴 The model predicts a HIGH likelihood of heart disease.")
else:
    print("🟢 The model predicts a LOW likelihood of heart disease.")
if probability is not None:
    print(f"📊 Confidence Score: {probability * 100:.2f}%")
print("=======================================\n")
