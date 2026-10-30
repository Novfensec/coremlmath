import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

from ._cmlmath import *  # type: ignore
from .tree import DecisionTreeClassifier, Node, export_text, plot_tree

__all__ = [
    "set_cpu_cores",
    "get_cpu_cores",
    "information_gain",
    "gini_gain",
    "DecisionTreeClassifier",
    "Node",
    "export_text",
    "plot_tree",
]
