import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

os.makedirs('models', exist_ok=True)

print("="*60)
print("STUDENT PERFORMANCE PREDICTION - TRAINING")
print("="*60)

print("\nCreating synthetic dataset...")
np.random.seed(42)
n_samples = 2000

df = pd.DataFrame({
    'age': np.random.randint(15, 23, n_samples),
    'G1': np.random.randint(0, 20, n_samples),
    'G2': np.random.randint(0, 20, n_samples),
    'failures': np.random.randint(0, 4, n_samples),
    'studytime': np.random.randint(1, 5, n_samples),
    'absences': np.random.randint(0, 30, n_samples),
    'sex': np.random.choice(['M', 'F'], n_samples),
    'internet': np.random.choice(['yes', 'no'], n_samples),
    'activities': np.random.choice(['yes', 'no'], n_samples),
})

df['G3'] = (df['G1'] + df['G2']) / 2 + np.random.normal(0, 2, n_samples)
df['G3'] = df['G3'].clip(0, 20).astype(int)

def classify(grade):
    if grade <= 10:
        return 0
    elif grade <= 15:
        return 1
    return 2

df['target'] = df['G3'].apply(classify)

print(f"Dataset: {len(df)} samples")
print(f"Classes: At Risk={(df['target']==0).sum()}, Moderate={(df['target']==1).sum()}, High={(df['target']==2).sum()}")

le = LabelEncoder()
for col in ['sex', 'internet', 'activities']:
    df[col] = le.fit_transform(df[col])

feature_cols = ['age', 'G1', 'G2', 'failures', 'studytime', 'absences', 'sex', 'internet', 'activities']
X = df[feature_cols]
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

print("\nSaving models...")
joblib.dump(lr, 'models/logistic_model.pkl')
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_cols, 'models/features.pkl')

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nResults:")
print(f"Logistic Regression: {lr_acc:.4f}")
print(f"Random Forest: {rf_acc:.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['At Risk', 'Moderate', 'High']))
print("\n✅ Models saved to 'models/' folder")
