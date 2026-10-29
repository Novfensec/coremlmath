import numpy as np
import cmlmath
import time

print("Generating massive dataset (1,000,000 rows, 500 features)...")
X_raw = np.random.randint(0, 2, size=(1000000, 500), dtype=np.int32)
y_raw = np.random.randint(0, 16, size=(1000000,), dtype=np.int32)

# Enforce C-contiguous (Row-Major) layout in memory. 
# (np.random generates C-contiguous arrays by default, but this guarantees 
# safety if you ever swap the data source out for a Pandas DataFrame or transposed array).
X = np.ascontiguousarray(X_raw, dtype=np.int32)
y = np.ascontiguousarray(y_raw, dtype=np.int32)

print(f"Starting multithreaded C++ calculation using {cmlmath.get_cpu_cores()} cores...")
start = time.time()

info_gains = cmlmath.information_gain(X, y)
gini_gains = cmlmath.gini_gain(X, y)

calc_time = time.time() - start

print(f"Calculated 500 Information and Gini Gains in {calc_time:.4f} seconds.")
print(f"Best feature index to split on (Information Gain): {np.argmax(info_gains)}")
print(f"Best feature index to split on (Gini Gain): {np.argmax(gini_gains)}")