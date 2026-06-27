import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ==============================================================================
# 1. LOAD THE DATASET
# ==============================================================================
print("Loading loan dataset...")
# Replace 'loan_data.csv' with your specific file name if it differs
df = pd.read_csv('loan_data.csv')

print(f"Dataset Dimensions: {df.shape}")

# ==============================================================================
# 2. DATA PREPROCESSING & CLEANING
# ==============================================================================
print("\nPreprocessing data...")

# A. Handle Missing Values
# Identify columns with missing values and fill them strategically
missing_vals = df.isnull().sum()
print("Missing values per column prior to cleaning:")
print(missing_vals[missing_vals > 0])

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            # Fill missing categorical data with the most frequent value (mode)
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # Fill missing numerical data with the median to avoid outlier skewing
            df[col] = df[col].fillna(df[col].median())

# B. Encode Categorical Variables
# Machine learning models only accept numbers. We convert text columns using Label Encoding.
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

# Assume our target variable is named 'loan_status' (e.g., 'Approved'/'Fully Paid' vs 'Charged Off'/'Default')
# Remove target column from categorical features if it's a string
target_col = 'loan_status' 

for col in categorical_cols:
    if col != target_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Ensure target column is binary (0 = Good/Paid, 1 = Default/Risk)
# Adjust these strings based on your exact Kaggle dataset labels
if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].map({'Fully Paid': 0, 'Approved': 0, 'Charged Off': 1, 'Default': 1})

# Drop any rows where target mapping failed or was missing originally
df = df.dropna(subset=[target_col])

# ==============================================================================
# 3. SPLIT DATA & FEATURE SCALING
# ==============================================================================
X = df.drop(target_col, axis=1)
y = df[target_col].astype(int)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features so features with large numbers (like Annual Income) 
# don't mathematically overwhelm smaller indicators (like Interest Rate)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 4. TRAIN THE RANDOM FOREST MODEL
# ==============================================================================
print("\nTraining Risk Assessment Model...")
# 'class_weight="balanced"' helps the trees handle any lingering imbalance in loan defaults
model = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# ==============================================================================
# 5. MODEL EVALUATION
# ==============================================================================
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n================ EVALUATION REPORT ================")
print(classification_report(y_test, y_pred, target_names=['Low Risk (Paid)', 'High Risk (Default)']))

# ==============================================================================
# 6. FEATURE IMPORTANCE (The Financial Insight)
# ==============================================================================
# One of the best perks of Random Forest is it tells us *why* it made decisions
importances = model.feature_importances()
indices = np.argsort(importances)[::-1]

print("\nTop 5 Most Critical Risk Factors:")
for i in range(5):
    print(f"{i+1}. Feature: {X.columns[indices[i]]} (Importance Score: {importances[indices[i]]:.4f})")

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices[:10]], y=X.columns[indices[:10]], palette="viridis")
plt.title('Top 10 Feature Importances for Loan Default Risk')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()
