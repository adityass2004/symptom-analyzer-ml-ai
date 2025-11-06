import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("DISEASE PREDICTION MODEL - COMPLETE TRAINING PIPELINE")
print("="*80)

# ====================================================
# 1. LOAD AND EXPLORE DATA
# ====================================================
print("\n[1] LOADING DATASETS...")
training_data = "dataset/Training.csv"
testing_data = "dataset/Testing.csv"

try:
    train_df = pd.read_csv(training_data)
    test_df = pd.read_csv(testing_data)
    print(f"✓ Training data loaded: {train_df.shape}")
    print(f"✓ Testing data loaded: {test_df.shape}")
except FileNotFoundError:
    print("❌ Dataset files not found. Please check the paths.")
    exit()

# ====================================================
# 2. DATA PREPROCESSING
# ====================================================
print("\n[2] DATA PREPROCESSING...")

# Remove any unnamed columns and NaN values
train_df = train_df.dropna(axis=1)
test_df = test_df.dropna(axis=1)

print(f"✓ After cleaning - Train: {train_df.shape}, Test: {test_df.shape}")
print(f"✓ Number of unique diseases: {train_df['prognosis'].nunique()}")
print(f"✓ Disease distribution:\n{train_df['prognosis'].value_counts().head()}")

# Encode target variable
encoder = LabelEncoder()
train_df['prognosis_encoded'] = encoder.fit_transform(train_df['prognosis'])
test_df['prognosis_encoded'] = encoder.transform(test_df['prognosis'])

# Split features and target
X_train = train_df.iloc[:, :-2]  # All columns except prognosis and encoded
y_train = train_df['prognosis_encoded']
X_test = test_df.iloc[:, :-2]
y_test = test_df['prognosis_encoded']

print(f"✓ Features: {X_train.shape[1]}")
print(f"✓ Training samples: {X_train.shape[0]}")
print(f"✓ Testing samples: {X_test.shape[0]}")

# ====================================================
# 3. BASELINE MODEL COMPARISON
# ====================================================
print("\n[3] COMPARING MULTIPLE ALGORITHMS...")

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Display results
results_df = pd.DataFrame(results).T
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY:")
print("="*80)
print(results_df)
print("\nBest Model:", results_df['Accuracy'].idxmax())

# ====================================================
# 4. HYPERPARAMETER TUNING FOR RANDOM FOREST
# ====================================================
print("\n[4] HYPERPARAMETER TUNING FOR RANDOM FOREST...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

print("Starting Grid Search (this may take a few minutes)...")
grid_search.fit(X_train, y_train)

print("\n✓ Best Parameters:", grid_search.best_params_)
print(f"✓ Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Use best model
best_rf_model = grid_search.best_estimator_

# ====================================================
# 5. DETAILED EVALUATION OF BEST MODEL
# ====================================================
print("\n[5] DETAILED EVALUATION OF OPTIMIZED RANDOM FOREST...")

# Predictions
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)

# Training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\n✓ Training Accuracy: {train_accuracy:.4f}")

# Testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print(f"✓ Testing Accuracy:  {test_accuracy:.4f}")
print(f"✓ Testing Precision: {test_precision:.4f}")
print(f"✓ Testing Recall:    {test_recall:.4f}")
print(f"✓ Testing F1-Score:  {test_f1:.4f}")

# Cross-validation scores
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=10, scoring='accuracy')
print(f"\n✓ 10-Fold Cross-Validation Scores: {cv_scores}")
print(f"✓ Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ====================================================
# 6. CLASSIFICATION REPORT
# ====================================================
print("\n[6] CLASSIFICATION REPORT:")
print("="*80)
print(classification_report(
    y_test, 
    y_test_pred, 
    target_names=encoder.classes_,
    zero_division=0
))

# ====================================================
# 7. FEATURE IMPORTANCE
# ====================================================
print("\n[7] TOP 20 MOST IMPORTANT FEATURES:")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(20))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 20 Most Important Symptoms for Disease Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance plot saved as 'feature_importance.png'")

# ====================================================
# 8. CONFUSION MATRIX
# ====================================================
print("\n[8] GENERATING CONFUSION MATRIX...")

cm = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix (simplified for many classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Disease Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
class_accuracy_df = pd.DataFrame({
    'Disease': encoder.classes_,
    'Accuracy': class_accuracy
}).sort_values('Accuracy', ascending=False)

print("\nPer-Class Accuracy (Top 10):")
print(class_accuracy_df.head(10))
print("\nPer-Class Accuracy (Bottom 10):")
print(class_accuracy_df.tail(10))

# ====================================================
# 9. MODEL PERFORMANCE VISUALIZATION
# ====================================================
print("\n[9] CREATING PERFORMANCE VISUALIZATIONS...")

# Accuracy comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model comparison
ax1 = axes[0, 0]
results_df.plot(kind='bar', ax=ax1, colormap='viridis')
ax1.set_title('Model Comparison - All Metrics')
ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.set_ylim([0, 1.1])

# 2. Cross-validation scores
ax2 = axes[0, 1]
ax2.plot(range(1, 11), cv_scores, marker='o', linestyle='-', color='green')
ax2.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax2.set_title('10-Fold Cross-Validation Scores')
ax2.set_xlabel('Fold')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

# 3. Train vs Test accuracy
ax3 = axes[1, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy, 
                precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                f1_score(y_train, y_train_pred, average='weighted', zero_division=0)]
test_scores = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35
ax3.bar(x - width/2, train_scores, width, label='Train', color='lightblue')
ax3.bar(x + width/2, test_scores, width, label='Test', color='lightcoral')
ax3.set_ylabel('Score')
ax3.set_title('Training vs Testing Performance')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.set_ylim([0, 1.1])

# 4. Top 10 feature importance
ax4 = axes[1, 1]
top_10_features = feature_importance.head(10)
ax4.barh(range(len(top_10_features)), top_10_features['Importance'], color='orange')
ax4.set_yticks(range(len(top_10_features)))
ax4.set_yticklabels(top_10_features['Feature'])
ax4.set_xlabel('Importance')
ax4.set_title('Top 10 Most Important Features')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('model_performance_summary.png', dpi=300, bbox_inches='tight')
print("✓ Performance summary saved as 'model_performance_summary.png'")

# ====================================================
# 10. SAVE MODEL AND ARTIFACTS
# ====================================================
print("\n[10] SAVING MODEL AND ARTIFACTS...")

# Save the trained model
joblib.dump(best_rf_model, 'disease_prediction_model.pkl')
print("✓ Model saved as 'disease_prediction_model.pkl'")

# Save the label encoder
joblib.dump(encoder, 'label_encoder.pkl')
print("✓ Label encoder saved as 'label_encoder.pkl'")

# Save feature names
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print("✓ Feature names saved as 'feature_names.pkl'")

# Create symptom index dictionary
symptom_index = {
    " ".join([i.capitalize() for i in value.split("_")]): index
    for index, value in enumerate(feature_names)
}
joblib.dump(symptom_index, 'symptom_index.pkl')
print("✓ Symptom index saved as 'symptom_index.pkl'")

# Save training summary
summary = {
    'model_type': 'Random Forest Classifier',
    'best_parameters': grid_search.best_params_,
    'training_accuracy': train_accuracy,
    'testing_accuracy': test_accuracy,
    'testing_precision': test_precision,
    'testing_recall': test_recall,
    'testing_f1': test_f1,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'num_features': len(feature_names),
    'num_classes': len(encoder.classes_),
    'class_names': encoder.classes_.tolist()
}

import json
with open('model_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
print("✓ Model summary saved as 'model_summary.json'")

# ====================================================
# 11. FINAL SUMMARY
# ====================================================
print("\n" + "="*80)
print("FINAL MODEL SUMMARY")
print("="*80)
print(f"Model Type:           Random Forest Classifier")
print(f"Best Parameters:      {grid_search.best_params_}")
print(f"Training Accuracy:    {train_accuracy:.4f}")
print(f"Testing Accuracy:     {test_accuracy:.4f}")
print(f"Testing Precision:    {test_precision:.4f}")
print(f"Testing Recall:       {test_recall:.4f}")
print(f"Testing F1-Score:     {test_f1:.4f}")
print(f"CV Mean Accuracy:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Number of Features:   {len(feature_names)}")
print(f"Number of Diseases:   {len(encoder.classes_)}")
print("="*80)

print("\n✅ TRAINING COMPLETE!")
print("\nFiles generated:")
print("  1. disease_prediction_model.pkl")
print("  2. label_encoder.pkl")
print("  3. feature_names.pkl")
print("  4. symptom_index.pkl")
print("  5. model_summary.json")
print("  6. feature_importance.png")
print("  7. confusion_matrix.png")
print("  8. model_performance_summary.png")

print("\n" + "="*80)