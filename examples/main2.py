import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


def calculate_entropy(target_series: pd.Series):
    """Calculates the entropy for a given target series."""
    if len(target_series) == 0:
        return 0
    probabilities = target_series.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_gini(target_series: pd.Series):
    """Calculates the Gini impurity for a given target series."""
    if len(target_series) == 0:
        return 0
    probabilities = target_series.value_counts(normalize=True)
    return 1.0 - np.sum(probabilities ** 2)

def calculate_information_gain(df_features, target, feature_name):
    """Calculates the Information Gain of a specific feature split."""
    base_entropy = calculate_entropy(target)

    left_split = target[df_features[feature_name] == 0]
    right_split = target[df_features[feature_name] == 1]

    weight_left = len(left_split) / len(target)
    weight_right = len(right_split) / len(target)

    weighted_entropy = (weight_left * calculate_entropy(left_split)) + \
                       (weight_right * calculate_entropy(right_split))

    return base_entropy - weighted_entropy

def calculate_gini_gain(df_features, target, feature_name):
    """Calculates the Gini Gain of a specific feature split."""
    base_gini = calculate_gini(target)

    left_split = target[df_features[feature_name] == 0]
    right_split = target[df_features[feature_name] == 1]

    weight_left = len(left_split) / len(target)
    weight_right = len(right_split) / len(target)

    weighted_gini = (weight_left * calculate_gini(left_split)) + \
                    (weight_right * calculate_gini(right_split))

    return base_gini - weighted_gini

# --- 2. Main Execution ---

df = pd.read_csv("characters.csv")

# Feature Engineering
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

le = LabelEncoder()
df['name_label'] = le.fit_transform(df['name'])

# Setup X and y using Pandas (no np.ascontiguousarray needed for pure Python)
feature_names = bool_cols.tolist()
X = df[feature_names]
y = df['name_label']

# Train full tree for visualization
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# --- 3. Calculate gains using Pure Python ---
start = time.time()

info_gains = []
gini_gains = []

for col in feature_names:
    info_gains.append(calculate_information_gain(X, y, col))
    gini_gains.append(calculate_gini_gain(X, y, col))

calc_time = time.time() - start

# --- 4. Map the Python output back to DataFrame ---
results_df = pd.DataFrame({
    'Feature': feature_names,
    'Information_Gain': info_gains,
    'Gini_Gain': gini_gains
}).sort_values(by='Information_Gain', ascending=False).reset_index(drop=True)

print(f"Calculated gains in {calc_time:.4f} seconds using Pure Python.\n")
print(results_df)

best_feature = results_df.iloc[0]['Feature']
print(f"\nBest feature to split on: {best_feature}")

# --- 5. Plot the Root Node ---
plt.figure(figsize=(8, 5))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=le.classes_.tolist(),
    filled=True,
    rounded=True,
    max_depth=0,
    fontsize=10
)
plt.title(f"Step 1: Root Node (Best split: {best_feature})")
plt.show()