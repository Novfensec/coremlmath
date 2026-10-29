import pandas as pd
import numpy as np
import cmlmath
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("characters.csv")
X = df.drop(columns=['name'])
y = df['name']

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

le = LabelEncoder()
df['name_label'] = le.fit_transform(df['name'])

feature_names = bool_cols.tolist()
X = np.ascontiguousarray(df[feature_names].values, dtype=np.int32)
y = np.ascontiguousarray(df['name_label'].values, dtype=np.int32)

start = time.time()
info_gains = cmlmath.information_gain(X, y)
gini_gains = cmlmath.gini_gain(X, y)
calc_time = time.time() - start

results_df = pd.DataFrame({
    'Feature': feature_names,
    'Information_Gain': info_gains,
    'Gini_Gain': gini_gains
}).sort_values(by='Information_Gain', ascending=False).reset_index(drop=True)

print(f"Calculated gains in {calc_time:.4f} seconds using {cmlmath.get_cpu_cores()} cores.\n")
print(results_df)

best_feature_idx = np.argmax(info_gains)
print(f"\nBest feature to split on: {feature_names[best_feature_idx]}")