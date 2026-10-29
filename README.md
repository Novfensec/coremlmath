# coremlmath

A high-performance C++ backend for Machine Learning math in Python.

> [!NOTE]
> This library uses OpenMP to bypass the Python GIL and multithread calculations across all available CPU cores.

Thanks to [pybind11](https://github.com/pybind/pybind11) and OpenMP for existing in this world.

## Financial Support
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Novfensec?style=for-the-badge&label=Sponsor%20Novfensec&logo=github&color=000000)](https://github.com/sponsors/Novfensec)

[![Donate via](https://img.shields.io/badge/Donate%20via-Wise-9FE870?style=for-the-badge&logo=wise&labelColor=163300)](https://wise.com/pay/business/kartavyashukla)

[![Donate via PayPal](https://img.shields.io/badge/Donate%20via-PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://www.paypal.me/KARTAVYASHUKLA)


## Build Instructions

- Clone the repo locally:

    ```sh
    git clone https://github.com/novfensec/coremlmath -b main --single-branch --depth 1
    cd coremlmath
    ```

- Linux:
    ```sh
    sudo apt update
    sudo apt install -y build-essential cmake
    ```

- Windows (Powershell):

    Install the necessary C++ build tools and CMake using Windows Package Manager (`winget`):
    > This has to be done only once.

    ```powershell
    winget install -e --id Microsoft.VisualStudio.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --source winget
    winget install -e --id Kitware.CMake --source winget
    ```

- Install pybind11:
    ```sh
    pip install "pybind11[global]"
    ```

- Build and install using standard pip:
    ```sh
    pip install .

    ```

    Or using `uv` for isolated, fast builds:

    ```sh
    uv pip install .
    ```

## Usage

The package expects 2D and 1D NumPy arrays of integers for maximum memory efficiency.

```python
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
```