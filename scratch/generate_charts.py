import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

# Create Charts directory if it doesn't exist
if not os.path.exists('Charts'):
    os.makedirs('Charts')

print("Generating charts quickly...")

# Load data
train_df = pd.read_csv('dataset/Training.csv')
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
train_df.columns = [col.strip().lower() for col in train_df.columns]

# 1. Disease Distribution
plt.figure(figsize=(15, 10))
sns.countplot(y='prognosis', data=train_df, order=train_df['prognosis'].value_counts().index, palette="viridis")
plt.title("Disease Frequency Distribution")
plt.tight_layout()
plt.savefig('Charts/disease_distribution.png')
print("Saved Charts/disease_distribution.png")

# 2. Top Features
X = train_df.drop('prognosis', axis=1)
y = train_df['prognosis']
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)

feature_importance_df = pd.DataFrame({
    'Symptom': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Symptom', data=feature_importance_df.head(20), palette="rocket")
plt.title("Top 20 Most Predictive Symptoms")
plt.tight_layout()
plt.savefig('Charts/top_features.png')
print("Saved Charts/top_features.png")

# 3. Confusion Matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("Confusion Matrix - Quick Reference")
plt.tight_layout()
plt.savefig('Charts/confusion_matrix_final.png')
print("Saved Charts/confusion_matrix_final.png")

print("All charts generated successfully!")
