import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('dataset/Training.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

diseases = df['prognosis'].unique()
symptom_cols = [col for col in df.columns if col != 'prognosis']

report = []

print("--- Dataset Audit: Disease-Symptom Correlations ---")
for disease in diseases:
    disease_df = df[df['prognosis'] == disease]
    total_samples = len(disease_df)
    
    # Calculate frequency of each symptom for this disease
    symptom_freq = disease_df[symptom_cols].sum() / total_samples
    top_symptoms = symptom_freq[symptom_freq > 0.5].index.tolist()
    
    report.append({
        'disease': disease,
        'top_symptoms': top_symptoms,
        'all_freqs': symptom_freq[symptom_freq > 0].to_dict()
    })
    
    # Flag suspicious correlations
    # Fever, Headache, Fatigue are very general. If they are the ONLY symptoms for something specific like Varicose Veins, it's noise.
    general_symptoms = ['high_fever', 'headache', 'fatigue', 'fever']
    found_general = [s for s in general_symptoms if s in top_symptoms]
    
    if disease in ['Varicose veins', 'Acne', 'Arthritis', 'Psoriasis'] and len(found_general) > 0:
        print(f"[SUSPICIOUS] {disease} has top symptoms: {top_symptoms}")

# Specific check for Varicose veins
vv_df = df[df['prognosis'] == 'Varicose veins']
print("\nVaricose veins symptom profile in dataset:")
print(vv_df[symptom_cols].sum()[vv_df[symptom_cols].sum() > 0])
