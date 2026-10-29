#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

double entropy_from_counts(const std::vector<int> &counts, int total_samples)
{
    if (total_samples == 0)
        return 0.0;
    double entropy = 0.0;
    for (int count : counts)
    {
        if (count > 0)
        {
            double prob = static_cast<double>(count) / total_samples;
            entropy -= prob * std::log2(prob);
        }
    }
    return entropy;
}

double gini_from_counts(const std::vector<int> &counts, int total_samples)
{
    if (total_samples == 0)
        return 0.0;
    double sum_squared_probs = 0.0;
    for (int count : counts)
    {
        if (count > 0)
        {
            double prob = static_cast<double>(count) / total_samples;
            sum_squared_probs += prob * prob;
        }
    }
    return 1.0 - sum_squared_probs;
}

void set_cpu_cores(int num_cores)
{
    if (num_cores > 0)
    {
        omp_set_num_threads(num_cores);
    }
}

int get_cpu_cores()
{
    return omp_get_max_threads();
}

py::array_t<double> information_gain(py::array_t<int> X, py::array_t<int> y)
{
    py::buffer_info X_buf = X.request();
    py::buffer_info y_buf = y.request();

    int *X_ptr = static_cast<int *>(X_buf.ptr);
    int *y_ptr = static_cast<int *>(y_buf.ptr);

    int num_samples = X_buf.shape[0];
    int num_features = X_buf.shape[1];

    auto result = py::array_t<double>(num_features);
    py::buffer_info res_buf = result.request();
    double *res_ptr = static_cast<double *>(res_buf.ptr);

    py::gil_scoped_release release;

    int max_label = 0;
    for (int i = 0; i < num_samples; i++)
    {
        if (y_ptr[i] > max_label)
            max_label = y_ptr[i];
    }
    int num_classes = max_label + 1;

    std::vector<int> base_counts(num_classes, 0);
    for (int i = 0; i < num_samples; i++)
    {
        base_counts[y_ptr[i]]++;
    }
    double base_entropy = entropy_from_counts(base_counts, num_samples);

#pragma omp parallel
    {
        std::vector<int> left_counts(num_classes);
        std::vector<int> right_counts(num_classes);

#pragma omp for
        for (int feat = 0; feat < num_features; feat++)
        {
            std::fill(left_counts.begin(), left_counts.end(), 0);
            std::fill(right_counts.begin(), right_counts.end(), 0);

            int left_total = 0;
            int right_total = 0;

            for (int i = 0; i < num_samples; i++)
            {
                int label = y_ptr[i];
                if (X_ptr[i * num_features + feat] == 0)
                {
                    left_counts[label]++;
                    left_total++;
                }
                else
                {
                    right_counts[label]++;
                    right_total++;
                }
            }

            double w_left = static_cast<double>(left_total) / num_samples;
            double w_right = static_cast<double>(right_total) / num_samples;

            double child_entropy = (w_left * entropy_from_counts(left_counts, left_total)) +
                                   (w_right * entropy_from_counts(right_counts, right_total));

            res_ptr[feat] = base_entropy - child_entropy;
        }
    }

    return result;
}

py::array_t<double> gini_gain(py::array_t<int> X, py::array_t<int> y)
{
    py::buffer_info X_buf = X.request();
    py::buffer_info y_buf = y.request();

    int *X_ptr = static_cast<int *>(X_buf.ptr);
    int *y_ptr = static_cast<int *>(y_buf.ptr);

    int num_samples = X_buf.shape[0];
    int num_features = X_buf.shape[1];

    auto result = py::array_t<double>(num_features);
    py::buffer_info res_buf = result.request();
    double *res_ptr = static_cast<double *>(res_buf.ptr);

    py::gil_scoped_release release;

    int max_label = 0;
    for (int i = 0; i < num_samples; i++)
    {
        if (y_ptr[i] > max_label)
            max_label = y_ptr[i];
    }
    int num_classes = max_label + 1;

    std::vector<int> base_counts(num_classes, 0);
    for (int i = 0; i < num_samples; i++)
    {
        base_counts[y_ptr[i]]++;
    }
    double base_gini = gini_from_counts(base_counts, num_samples);

#pragma omp parallel
    {
        std::vector<int> left_counts(num_classes);
        std::vector<int> right_counts(num_classes);

#pragma omp for
        for (int feat = 0; feat < num_features; feat++)
        {
            std::fill(left_counts.begin(), left_counts.end(), 0);
            std::fill(right_counts.begin(), right_counts.end(), 0);

            int left_total = 0;
            int right_total = 0;

            for (int i = 0; i < num_samples; i++)
            {
                int label = y_ptr[i];
                if (X_ptr[i * num_features + feat] == 0)
                {
                    left_counts[label]++;
                    left_total++;
                }
                else
                {
                    right_counts[label]++;
                    right_total++;
                }
            }

            double w_left = static_cast<double>(left_total) / num_samples;
            double w_right = static_cast<double>(right_total) / num_samples;

            double child_gini = (w_left * gini_from_counts(left_counts, left_total)) +
                                (w_right * gini_from_counts(right_counts, right_total));

            res_ptr[feat] = base_gini - child_gini;
        }
    }

    return result;
}

PYBIND11_MODULE(_cmlmath, m)
{
    m.doc() = "CoreMLMath: Multithreaded C++ backend for Machine Learning Math";

    m.def("set_cpu_cores", &set_cpu_cores,
          "Set the number of CPU cores (threads) used for parallel calculations.",
          py::arg("num_cores"));

    m.def("get_cpu_cores", &get_cpu_cores,
          "Get the current maximum number of CPU cores available for calculations.");

    m.def("information_gain", &information_gain,
          "Calculates Information Gain (Entropy) for a 2D feature matrix in parallel",
          py::arg("X"), py::arg("y"));

    m.def("gini_gain", &gini_gain,
          "Calculates Gini Gain for a 2D feature matrix in parallel",
          py::arg("X"), py::arg("y"));
}