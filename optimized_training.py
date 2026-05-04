import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)

def audit_and_clean_dataset(df):
    """
    Audits the dataset for medically inconsistent mappings and removes noisy samples.
    """
    print("--- [1] Medical Audit & Noisy Sample Removal ---")
    
    # 1. Basic Cleaning
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Handle duplicate column names
    cols_to_remove = []
    seen_cols = {}
    for col in df.columns:
        clean_name = col.split('.')[0].strip().lower()
        if clean_name in seen_cols:
            cols_to_remove.append(col)
        else:
            seen_cols[clean_name] = col
    df = df.drop(columns=cols_to_remove)
    
    # 2. Medical Consistency Audit
    # Define "General Symptoms" that shouldn't be the primary indicators for localized/specific diseases
    general_symptoms = ['high_fever', 'headache', 'fatigue', 'fever', 'nausea']
    specific_diseases = ['Varicose veins', 'Acne', 'Arthritis', 'Psoriasis', 'Hypertension ', 'Diabetes ']
    
    initial_shape = df.shape
    
    # We want to identify if a sample for a 'specific disease' has ONLY general symptoms 
    # or lacks its 'core symptoms'.
    # For now, let's just identify and remove samples that associate high-fever/headache with localized diseases.
    
    inconsistent_mask = (
        (df['prognosis'].isin(specific_diseases)) & 
        ((df['high_fever'] == 1) | (df['headache'] == 1))
    )
    
    noisy_count = inconsistent_mask.sum()
    if noisy_count > 0:
        print(f"  - Flagged {noisy_count} medically inconsistent samples (e.g., Fever/Headache in Localized Diseases)")
        df = df[~inconsistent_mask]
        
    # Check for overlapping symptom profiles that cause confusion
    # If a sample has symptoms that perfectly match another disease better, it might be noise
    # (But that's harder to automate without a ground truth medical knowledge base)
    
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        
    # Validate binary encoding
    symptom_cols = df.columns[:-1]
    for col in symptom_cols:
        if not set(df[col].unique()).issubset({0, 1}):
            df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)
            
    print(f"  [OK] Dataset cleaned. Samples: {initial_shape[0]} -> {df.shape[0]}")
    return df

def print_disease_profiles(model, feature_names, encoder):
    """Prints the top 10 most important symptoms for each disease."""
    print("\n--- Disease Core Symptom Profiles ---")
    
    # For Random Forest, we can look at feature importances globally, 
    # but for per-class importances we need a different approach or just a summary.
    # However, we can calculate the mean symptom presence per class in the training data.
    pass # We will implement this in the main loop or as a separate analysis

def analyze_data(df):
    """Analysis of class distribution and imbalance."""
    print("\n--- [2] Data Analysis ---")
    
    target = 'prognosis'
    class_counts = df[target].value_counts()
    
    print(f"  - Total Classes: {len(class_counts)}")
    print(f"  - Imbalance Ratio: {class_counts.max() / class_counts.min():.2f}")
    
    plt.figure(figsize=(15, 10))
    sns.countplot(y=target, data=df, order=class_counts.index, palette="viridis")
    plt.title("Disease Frequency Distribution")
    plt.tight_layout()
    plt.savefig('Charts/disease_distribution.png')
    print("  [OK] Saved distribution plot to Charts/disease_distribution.png")

def engineer_features(X, y):
    """Feature selection using Random Forest importance."""
    print("\n--- [3] Feature Selection ---")
    
    selector_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector_rf.fit(X, y)
    
    feature_importance_df = pd.DataFrame({
        'Symptom': X.columns,
        'Importance': selector_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Use median threshold for feature selection
    selection = SelectFromModel(selector_rf, threshold='median', prefit=True)
    X_selected = X.loc[:, selection.get_support()]
    
    print(f"  - Selected {X_selected.shape[1]} strongest symptoms (from {X.shape[1]})")
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Symptom', data=feature_importance_df.head(20), palette="rocket")
    plt.title("Top 20 Most Predictive Symptoms")
    plt.tight_layout()
    plt.savefig('Charts/top_features.png')
    
    return X_selected, selection

def train_and_optimize(X, y):
    """Train and compare models with calibration."""
    print("\n--- [4] Training & Optimization ---")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models_config = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1],
                'max_depth': [3, 5]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
    }
    
    best_models = {}
    cv_results = {}
    best_grids = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, config in models_config.items():
        print(f"  - Tuning {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=skf, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        cv_results[name] = grid.best_score_
        best_grids[name] = grid
        print(f"    Best CV Accuracy: {grid.best_score_:.4f}")
        
    best_model_name = max(cv_results, key=cv_results.get)
    print(f"\n  [OK] Overall Best Model: {best_model_name}")
    
    best_grid = best_grids[best_model_name]
    
    # Get training accuracy for the best model
    train_acc = accuracy_score(y_train, best_models[best_model_name].predict(X_train))
    
    # Get CV std
    best_idx = best_grid.best_index_
    cv_std = best_grid.cv_results_['std_test_score'][best_idx]
    
    print("\n--- [5] Probability Calibration ---")
    calibrated_model = CalibratedClassifierCV(
        estimator=best_models[best_model_name],
        method='sigmoid', 
        cv='prefit'
    )
    calibrated_model.fit(X_val, y_val)
    
    model_info = {
        'model_name': best_model_name,
        'cv_mean': cv_results[best_model_name],
        'cv_std': cv_std,
        'training_accuracy': train_acc,
        'best_params': best_grid.best_params_
    }
    
    return calibrated_model, X_train, X_val, y_train, y_val, model_info

def evaluate_model(model, X_test, y_test, encoder):
    """Final performance evaluation."""
    print("\n--- [6] Model Evaluation ---")
    
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    print(f"  - Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  - F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    print("\n  - Top-3 Prediction Examples:")
    for i in range(min(5, len(X_test))):
        sample_probs = probs[i]
        top3_idx = np.argsort(sample_probs)[-3:][::-1]
        top3_diseases = [(encoder.classes_[idx], sample_probs[idx]) for idx in top3_idx]
        print(f"    Test Case {i+1}: {', '.join([f'{d}: {p*100:.1f}%' for d, p in top3_diseases])}")
        
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title("Confusion Matrix - Final Calibrated Model")
    plt.savefig('Charts/confusion_matrix_final.png')
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(
            y_test, 
            y_pred, 
            labels=range(len(encoder.classes_)), 
            target_names=encoder.classes_, 
            output_dict=True
        )
    }

def main():
    if not os.path.exists('Charts'):
        os.makedirs('Charts')
        
    print("="*80)
    print("OPTIMIZED DISEASE PREDICTION TRAINING PIPELINE (V3 - MEDICAL AUDIT)")
    print("="*80)
        
    try:
        train_df = pd.read_csv('dataset/Training.csv')
        test_df = pd.read_csv('dataset/Testing.csv')
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Audit and Clean
    train_df = audit_and_clean_dataset(train_df)
    test_df = audit_and_clean_dataset(test_df)
    
    # Ensure test set only contains diseases present in training set after audit
    train_diseases = train_df['prognosis'].unique()
    test_df = test_df[test_df['prognosis'].isin(train_diseases)]
    
    # Calculate Core Symptom Profiles (for rule-based validation)
    print("\n--- [2] Generating Core Symptom Profiles ---")
    core_profiles = {}
    for disease in train_diseases:
        disease_df = train_df[train_df['prognosis'] == disease]
        symptom_freq = disease_df.drop('prognosis', axis=1).mean()
        top_10 = symptom_freq.sort_values(ascending=False).head(10)
        core_symptoms = top_10[top_10 > 0.1].index.tolist() # Keep symptoms present in at least 10%
        core_profiles[disease] = core_symptoms
        
        # Print Top 5 for user
        print(f"  - {disease}: {', '.join(top_10.index[:5])}...")

    analyze_data(train_df)
    
    encoder = LabelEncoder()
    train_df['prognosis'] = encoder.fit_transform(train_df['prognosis'])
    test_df['prognosis'] = encoder.transform(test_df['prognosis'])
    
    X = train_df.drop('prognosis', axis=1)
    y = train_df['prognosis']
    
    X_selected, selection_model = engineer_features(X, y)
    model, _, _, _, _, model_info = train_and_optimize(X_selected, y)
    
    X_test_real = test_df.drop('prognosis', axis=1).loc[:, selection_model.get_support()]
    y_test_real = test_df['prognosis']
    
    evaluation_metrics = evaluate_model(model, X_test_real, y_test_real, encoder)
    
    print("\n--- [7] Saving Artifacts ---")
    joblib.dump(model, 'disease_prediction_model.pkl')
    joblib.dump(encoder, 'label_encoder.pkl')
    
    feature_names = X_selected.columns.tolist()
    joblib.dump(feature_names, 'feature_names.pkl')
    
    symptom_index = {
        " ".join([i.capitalize() for i in value.split("_")]): index
        for index, value in enumerate(feature_names)
    }
    joblib.dump(symptom_index, 'symptom_index.pkl')
    
    # Save Core Profiles for app.py
    joblib.dump(core_profiles, 'core_profiles.pkl')
    print("  [OK] Saved core_profiles.pkl for rule-based validation.")
    
    summary = {
        'model_type': model_info['model_name'],
        'training_accuracy': model_info['training_accuracy'],
        'testing_accuracy': evaluation_metrics['accuracy'],
        'testing_precision': evaluation_metrics['report']['weighted avg']['precision'],
        'testing_recall': evaluation_metrics['report']['weighted avg']['recall'],
        'testing_f1': evaluation_metrics['report']['weighted avg']['f1-score'],
        'cv_mean': model_info['cv_mean'],
        'cv_std': model_info['cv_std'],
        'num_features': len(feature_names),
        'num_classes': len(encoder.classes_),
        'best_parameters': model_info['best_params'],
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('optimized_model_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"  [OK] Model saved. Final Accuracy: {evaluation_metrics['accuracy']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
