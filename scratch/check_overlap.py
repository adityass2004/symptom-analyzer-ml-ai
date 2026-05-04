import pandas as pd
df = pd.read_csv('dataset/Training.csv')
query = ['high_fever', 'headache', 'fatigue']
matches = df[(df[query] == 1).all(axis=1)]
print(f"Diseases with {query}:")
print(matches['prognosis'].value_counts())
