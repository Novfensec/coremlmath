"""
Decision Tree Classifier powered by the CoreMLMath multithreaded C++ backend.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from . import _cmlmath


class Node:
    """Represents an internal decision node or leaf node in the Decision Tree."""

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        feature_name: Optional[str] = None,
        threshold: Optional[float] = None,
        is_binary: bool = False,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        gain: float = 0.0,
        impurity: float = 0.0,
        samples: int = 0,
        value: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        predicted_class_idx: Optional[int] = None,
        predicted_class: Any = None,
        depth: int = 0,
        is_leaf: bool = False,
    ) -> None:
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.threshold = threshold
        self.is_binary = is_binary
        self.left = left
        self.right = right
        self.gain = gain
        self.impurity = impurity
        self.samples = samples
        self.value = value
        self.probabilities = probabilities
        self.predicted_class_idx = predicted_class_idx
        self.predicted_class = predicted_class
        self.depth = depth
        self.is_leaf = is_leaf

    def __repr__(self) -> str:
        if self.is_leaf:
            return (
                f"Leaf(class={self.predicted_class!r}, "
                f"samples={self.samples}, "
                f"impurity={abs(self.impurity):.4f})"
            )
        condition = (
            f"{self.feature_name} <= 0.5"
            if self.is_binary
            else f"{self.feature_name} <= {self.threshold:.4f}"
        )
        return (
            f"Node(split='{condition}', "
            f"gain={self.gain:.4f}, "
            f"samples={self.samples}, "
            f"impurity={abs(self.impurity):.4f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert node and subtree into a serializable dictionary."""
        if self.is_leaf:
            return {
                "type": "leaf",
                "predicted_class": (
                    self.predicted_class.item()
                    if hasattr(self.predicted_class, "item")
                    else self.predicted_class
                ),
                "samples": self.samples,
                "impurity": float(abs(self.impurity)),
                "value": self.value.tolist() if self.value is not None else None,
                "probabilities": (
                    self.probabilities.tolist()
                    if self.probabilities is not None
                    else None
                ),
                "depth": self.depth,
            }
        return {
            "type": "node",
            "feature_idx": self.feature_idx,
            "feature_name": self.feature_name,
            "threshold": self.threshold,
            "is_binary": self.is_binary,
            "gain": float(self.gain),
            "impurity": float(abs(self.impurity)),
            "samples": self.samples,
            "depth": self.depth,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }


def _compute_impurity(class_counts: np.ndarray, criterion: str) -> float:
    """Compute Entropy or Gini impurity for an array of class counts."""
    total = int(np.sum(class_counts))
    if total == 0:
        return 0.0
    probs = class_counts[class_counts > 0] / total
    if criterion in ("entropy", "information_gain"):
        entropy = -np.sum(probs * np.log2(probs))
        return float(abs(entropy))
    else:  # gini
        gini = 1.0 - np.sum(probs**2)
        return float(abs(gini))


class DecisionTreeClassifier:
    """A Decision Tree Classifier using CoreMLMath.

    This classifier utilizes CoreMLMath's OpenMP C++ backend to evaluate
    Information Gain (Entropy) and Gini Gain in parallel across CPU cores,
    bypassing Python's Global Interpreter Lock (GIL).

    Parameters
    ----------
    criterion : {"entropy", "gini"}, default="entropy"
        The function to measure the quality of a split. Supported criteria are
        "entropy" (or "information_gain") and "gini" (or "gini_gain").
    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until all
        leaves are pure or until all leaves contain less than min_samples_split.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        The number of features to consider when looking for the best split.
    max_bins : int, default=256
        Maximum number of candidate threshold bins for continuous features.
    tie_break : {"alphabetical", "first", "random"}, default="alphabetical"
        Strategy to resolve ties when two or more features produce identical
        maximum Information Gain / Entropy decrease.
        - "alphabetical": Chose the feature that comes first alphabetically.
        - "first": Chose the first feature by column index.
        - "random": Break ties pseudo-randomly using random_state.
    n_jobs : int or None, default=None
        Number of CPU threads to use for parallel calculations. If None,
        uses CoreMLMath's default (all available cores).
    random_state : int or None, default=None
        Controls feature subsampling and tie-breaking reproducibility.
    """

    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        max_bins: int = 256,
        tie_break: str = "alphabetical",
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if criterion not in ("entropy", "information_gain", "gini", "gini_gain"):
            raise ValueError(
                f"Invalid criterion '{criterion}'. Supported options are 'entropy' and 'gini'."
            )
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if tie_break not in ("alphabetical", "first", "random"):
            raise ValueError(
                f"Invalid tie_break '{tie_break}'. Supported options are 'alphabetical', 'first', and 'random'."
            )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_bins = max_bins
        self.tie_break = tie_break
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Fitted attributes
        self.tree_: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.feature_names_in_: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.max_depth_: int = 0

        self._is_binary_feature: List[bool] = []
        self._rng = np.random.default_rng(self.random_state)
        self._n_total_samples: int = 0

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "DecisionTreeClassifier":
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            The training input samples. Supports boolean, integer, float, and
            pandas DataFrame inputs.
        y : array-like or pandas.Series of shape (n_samples,)
            The target values (class labels). Can be strings, integers, or objects.
        feature_names : list of str, optional
            Names of the features. If X is a DataFrame, column names are used by default.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        if self.n_jobs is not None:
            _cmlmath.set_cpu_cores(int(self.n_jobs))

        if hasattr(X, "columns"):
            self.feature_names_in_ = list(str(c) for c in X.columns)
            X_mat = np.asarray(X.values)
        else:
            X_mat = np.asarray(X)
            if feature_names is not None:
                if len(feature_names) != X_mat.shape[1]:
                    raise ValueError(
                        f"Length of feature_names ({len(feature_names)}) does not match "
                        f"number of features in X ({X_mat.shape[1]})."
                    )
                self.feature_names_in_ = list(str(f) for f in feature_names)
            else:
                self.feature_names_in_ = [f"x{i}" for i in range(X_mat.shape[1])]

        if X_mat.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X_mat.shape}.")
        if X_mat.shape[0] == 0 or X_mat.shape[1] == 0:
            raise ValueError("X must contain at least 1 sample and 1 feature.")

        # Handle boolean or object arrays
        if X_mat.dtype == bool:
            X_mat = X_mat.astype(np.int32)
        elif X_mat.dtype.kind in ("O", "U", "S"):
            converted = []
            for col_idx in range(X_mat.shape[1]):
                col = X_mat[:, col_idx]
                try:
                    lower = np.char.lower(col.astype(str))
                    if set(np.unique(lower)).issubset(
                        {"true", "false", "0", "1", "yes", "no"}
                    ):
                        bool_vals = np.isin(lower, ["true", "1", "yes"]).astype(
                            np.int32
                        )
                        converted.append(bool_vals)
                    else:
                        converted.append(col.astype(float))
                except Exception:
                    _, codes = np.unique(col, return_inverse=True)
                    converted.append(codes.astype(np.int32))
            X_mat = np.column_stack(converted)

        # Handle y target
        if hasattr(y, "values"):
            y_raw = np.asarray(y.values)
        else:
            y_raw = np.asarray(y)

        if y_raw.ndim != 1:
            y_raw = y_raw.ravel()

        if len(y_raw) != X_mat.shape[0]:
            raise ValueError(
                f"Number of labels in y ({len(y_raw)}) does not match number of samples in X ({X_mat.shape[0]})."
            )

        # Encode target classes to 0 .. n_classes - 1
        self.classes_, y_encoded = np.unique(y_raw, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        y_encoded = y_encoded.astype(np.int32)

        self.n_features_in_ = X_mat.shape[1]
        self._n_total_samples = X_mat.shape[0]
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)

        # Re-initialize RNG with random_state for reproducible tie-breaking
        self._rng = np.random.default_rng(self.random_state)

        # Identify which features are strictly binary (0 and 1)
        self._is_binary_feature = []
        for feat_idx in range(self.n_features_in_):
            col = X_mat[:, feat_idx]
            u_vals = np.unique(col)
            is_bin = len(u_vals) <= 2 and set(u_vals).issubset({0, 1, 0.0, 1.0})
            self._is_binary_feature.append(is_bin)

        # Pre-convert X to contiguous float64 for consistent numeric thresholding
        X_clean = np.ascontiguousarray(X_mat, dtype=np.float64)

        # Build tree recursively
        all_indices = np.arange(self._n_total_samples, dtype=np.int32)
        self.max_depth_ = 0
        self.tree_ = self._build_tree(X_clean, y_encoded, all_indices, depth=0)

        # Normalize feature importances
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0.0:
            self.feature_importances_ /= total_imp

        return self

    def _select_features(self) -> np.ndarray:
        """Select a subset of feature indices."""
        n_features = self.n_features_in_
        if self.max_features is None:
            return np.arange(n_features)
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            k = max(1, int(math.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(math.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features value: {self.max_features}")
        return self._rng.choice(n_features, size=k, replace=False)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_indices: np.ndarray,
        depth: int,
    ) -> Node:
        """Recursively constructs the decision tree."""
        n_samples = len(sample_indices)
        if depth > self.max_depth_:
            self.max_depth_ = depth

        # Compute class distribution at this node
        y_node = y[sample_indices]
        class_counts = np.bincount(y_node, minlength=self.n_classes_)
        node_impurity = _compute_impurity(class_counts, self.criterion)
        pred_class_idx = int(np.argmax(class_counts))
        pred_class = self.classes_[pred_class_idx]
        probabilities = (
            class_counts / float(n_samples)
            if n_samples > 0
            else np.zeros(self.n_classes_)
        )

        # Check stopping criteria for leaf creation
        is_pure = len(np.unique(y_node)) <= 1
        depth_exceeded = self.max_depth is not None and depth >= self.max_depth
        too_few_samples = n_samples < self.min_samples_split

        if is_pure or depth_exceeded or too_few_samples:
            return Node(
                gain=0.0,
                impurity=node_impurity,
                samples=n_samples,
                value=class_counts,
                probabilities=probabilities,
                predicted_class_idx=pred_class_idx,
                predicted_class=pred_class,
                depth=depth,
                is_leaf=True,
            )

        # Build candidate splits for cmlmath
        features_to_check = self._select_features()
        cand_cols: List[np.ndarray] = []
        cand_meta: List[Tuple[int, float, bool]] = (
            []
        )  # (feature_idx, threshold, is_binary)

        X_node = X[sample_indices]

        # Fast path: check if all selected features are binary and in default natural order
        all_selected_binary = all(self._is_binary_feature[f] for f in features_to_check)

        if (
            all_selected_binary
            and len(features_to_check) == self.n_features_in_
            and np.array_equal(features_to_check, np.arange(self.n_features_in_))
        ):
            cand_matrix = np.ascontiguousarray(X_node, dtype=np.int32)
            cand_meta = [(f, 0.5, True) for f in range(self.n_features_in_)]
        else:
            for feat_idx in features_to_check:
                col = X_node[:, feat_idx]
                if self._is_binary_feature[feat_idx]:
                    cand_cols.append(col.astype(np.int32))
                    cand_meta.append((feat_idx, 0.5, True))
                else:
                    unique_vals = np.unique(col)
                    if len(unique_vals) <= 1:
                        continue
                    if len(unique_vals) - 1 > self.max_bins:
                        quantiles = np.linspace(0, 100, self.max_bins + 2)[1:-1]
                        thresholds = np.unique(np.percentile(unique_vals, quantiles))
                    else:
                        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

                    for thresh in thresholds:
                        cand_col = (col > thresh).astype(np.int32)
                        cand_cols.append(cand_col)
                        cand_meta.append((feat_idx, float(thresh), False))

            if not cand_cols:
                return Node(
                    gain=0.0,
                    impurity=node_impurity,
                    samples=n_samples,
                    value=class_counts,
                    probabilities=probabilities,
                    predicted_class_idx=pred_class_idx,
                    predicted_class=pred_class,
                    depth=depth,
                    is_leaf=True,
                )

            cand_matrix = np.ascontiguousarray(
                np.column_stack(cand_cols), dtype=np.int32
            )

        # Re-index local classes to compact 0 .. K_local - 1 for cmlmath memory efficiency
        _, y_local = np.unique(y_node, return_inverse=True)
        y_local = np.ascontiguousarray(y_local, dtype=np.int32)

        # Call CoreMLMath C++ multithreaded backend
        if self.criterion in ("entropy", "information_gain"):
            gains = _cmlmath.information_gain(cand_matrix, y_local)
        else:
            gains = _cmlmath.gini_gain(cand_matrix, y_local)

        best_cand_idx = -1
        best_gain = -1.0

        max_g = float(np.max(gains))
        if max_g > self.min_impurity_decrease and max_g > 0.0 and not math.isnan(max_g):
            # Find all candidates with the maximum gain (within numerical tolerance)
            tied_indices = [
                i for i, g in enumerate(gains) if np.isclose(g, max_g, atol=1e-7)
            ]

            # Break ties according to user-selected strategy
            if self.tie_break == "alphabetical":
                # Choose the one that comes first in alphabetical order of feature name
                tied_indices.sort(
                    key=lambda idx: (
                        self.feature_names_in_[cand_meta[idx][0]],
                        cand_meta[idx][1],
                    )
                )
            elif self.tie_break == "random" and self.random_state is not None:
                self._rng.shuffle(tied_indices)
            # If "first", original index order is preserved

            for cand_idx in tied_indices:
                right_count = int(np.sum(cand_matrix[:, cand_idx]))
                left_count = n_samples - right_count
                if (
                    left_count >= self.min_samples_leaf
                    and right_count >= self.min_samples_leaf
                ):
                    best_cand_idx = cand_idx
                    best_gain = max_g
                    break

        # Fallback if min_samples_leaf constraint needs scanning remaining candidates
        if best_cand_idx == -1 and self.min_samples_leaf > 1:
            sorted_candidates = np.argsort(gains)[::-1]
            for cand_idx in sorted_candidates:
                g = float(gains[cand_idx])
                if g <= self.min_impurity_decrease or g <= 0.0 or math.isnan(g):
                    break
                right_count = int(np.sum(cand_matrix[:, cand_idx]))
                left_count = n_samples - right_count
                if (
                    left_count >= self.min_samples_leaf
                    and right_count >= self.min_samples_leaf
                ):
                    best_cand_idx = int(cand_idx)
                    best_gain = g
                    break

        if best_cand_idx == -1:
            return Node(
                gain=0.0,
                impurity=node_impurity,
                samples=n_samples,
                value=class_counts,
                probabilities=probabilities,
                predicted_class_idx=pred_class_idx,
                predicted_class=pred_class,
                depth=depth,
                is_leaf=True,
            )

        best_feat_idx, best_threshold, is_binary = cand_meta[best_cand_idx]
        feat_name = self.feature_names_in_[best_feat_idx]

        # Partition samples
        # Note: in cmlmath, indicator == 0 goes left (<= threshold), indicator != 0 goes right (> threshold).
        col_vals = X_node[:, best_feat_idx]
        left_mask = col_vals <= best_threshold
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return Node(
                gain=0.0,
                impurity=node_impurity,
                samples=n_samples,
                value=class_counts,
                probabilities=probabilities,
                predicted_class_idx=pred_class_idx,
                predicted_class=pred_class,
                depth=depth,
                is_leaf=True,
            )

        # Accumulate feature importance
        weight = float(n_samples) / float(self._n_total_samples)
        self.feature_importances_[best_feat_idx] += weight * best_gain

        left_node = self._build_tree(X, y, sample_indices[left_mask], depth + 1)
        right_node = self._build_tree(X, y, sample_indices[right_mask], depth + 1)

        return Node(
            feature_idx=best_feat_idx,
            feature_name=feat_name,
            threshold=best_threshold,
            is_binary=is_binary,
            left=left_node,
            right=right_node,
            gain=best_gain,
            impurity=node_impurity,
            samples=n_samples,
            value=class_counts,
            probabilities=probabilities,
            predicted_class_idx=pred_class_idx,
            predicted_class=pred_class,
            depth=depth,
            is_leaf=False,
        )

    def _prepare_X(self, X: Any) -> np.ndarray:
        """Standardize and validate input X for inference."""
        if self.tree_ is None:
            raise RuntimeError(
                "This DecisionTreeClassifier instance is not fitted yet. Call 'fit' first."
            )

        if hasattr(X, "columns"):
            df_cols = list(str(c) for c in X.columns)
            if set(self.feature_names_in_).issubset(set(df_cols)):
                X_mat = np.asarray(X[self.feature_names_in_].values)
            else:
                X_mat = np.asarray(X.values)
        else:
            X_mat = np.asarray(X)

        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)

        if X_mat.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_mat.shape[1]} features, but DecisionTreeClassifier is expecting {self.n_features_in_} features."
            )

        if X_mat.dtype == bool:
            X_mat = X_mat.astype(np.float64)
        elif X_mat.dtype.kind in ("O", "U", "S"):
            converted = []
            for col_idx in range(X_mat.shape[1]):
                col = X_mat[:, col_idx]
                try:
                    lower = np.char.lower(col.astype(str))
                    if set(np.unique(lower)).issubset(
                        {"true", "false", "0", "1", "yes", "no"}
                    ):
                        converted.append(
                            np.isin(lower, ["true", "1", "yes"]).astype(float)
                        )
                    else:
                        converted.append(col.astype(float))
                except Exception:
                    _, codes = np.unique(col, return_inverse=True)
                    converted.append(codes.astype(float))
            X_mat = np.column_stack(converted)
        else:
            X_mat = X_mat.astype(np.float64)

        return X_mat

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Input test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (in original label space).
        """
        X_mat = self._prepare_X(X)
        n_samples = X_mat.shape[0]

        predictions = []
        for i in range(n_samples):
            curr = self.tree_
            row = X_mat[i]
            while not curr.is_leaf:
                val = row[curr.feature_idx]
                if val <= curr.threshold:
                    curr = curr.left
                else:
                    curr = curr.right
            predictions.append(curr.predicted_class)

        return np.array(predictions)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Input test samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        X_mat = self._prepare_X(X)
        n_samples = X_mat.shape[0]

        probas = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        for i in range(n_samples):
            curr = self.tree_
            row = X_mat[i]
            while not curr.is_leaf:
                val = row[curr.feature_idx]
                if val <= curr.threshold:
                    curr = curr.left
                else:
                    curr = curr.right
            probas[i] = curr.probabilities

        return probas

    def score(self, X: Any, y: Any) -> float:
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        if hasattr(y, "values"):
            y_true = np.asarray(y.values)
        else:
            y_true = np.asarray(y)
        if y_true.ndim != 1:
            y_true = y_true.ravel()

        return float(np.mean(y_pred == y_true))

    def get_depth(self) -> int:
        """Return the maximum depth of the decision tree."""
        if self.tree_ is None:
            return 0
        return self._node_depth(self.tree_)

    def _node_depth(self, node: Node) -> int:
        if node.is_leaf:
            return 0
        left_d = self._node_depth(node.left) if node.left else 0
        right_d = self._node_depth(node.right) if node.right else 0
        return 1 + max(left_d, right_d)

    def get_n_leaves(self) -> int:
        """Return the total number of leaves in the decision tree."""
        if self.tree_ is None:
            return 0
        return self._count_leaves(self.tree_)

    def _count_leaves(self, node: Node) -> int:
        if node.is_leaf:
            return 1
        count = 0
        if node.left:
            count += self._count_leaves(node.left)
        if node.right:
            count += self._count_leaves(node.right)
        return count

    def export_text(
        self,
        max_depth: Optional[int] = None,
        show_weights: bool = True,
    ) -> str:
        """Build a text representation of the decision tree."""
        if self.tree_ is None:
            return "Tree is empty (not fitted yet)."
        return export_text(self, max_depth=max_depth, show_weights=show_weights)

    def print_tree(self, max_depth: Optional[int] = None) -> None:
        """Print the decision tree to stdout."""
        print(self.export_text(max_depth=max_depth))

    def plot_tree(
        self,
        ax: Optional[Any] = None,
        max_depth: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        filled: bool = True,
        rounded: bool = True,
        precision: int = 1,
        fontsize: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 200,
        title: Optional[str] = "Step 4: Full Decision Tree (All Leaf Nodes Isolated)",
        save_path: Optional[Union[str, os.PathLike]] = None,
    ) -> Any:
        """Plot the decision tree in a large, collision-free, high-resolution layout."""
        res_ax = plot_tree(
            self,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            filled=filled,
            rounded=rounded,
            precision=precision,
            ax=ax,
            fontsize=fontsize,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )
        if save_path is not None:
            fig = res_ax.get_figure()
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        return res_ax

    def to_dict(self) -> Dict[str, Any]:
        """Convert the decision tree to a dictionary representation."""
        if self.tree_ is None:
            return {}
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "n_classes": self.n_classes_,
            "classes": [c.item() if hasattr(c, "item") else c for c in self.classes_],
            "feature_names": self.feature_names_in_,
            "tree": self.tree_.to_dict(),
        }


def export_text(
    decision_tree: DecisionTreeClassifier,
    feature_names: Optional[Sequence[str]] = None,
    max_depth: Optional[int] = None,
    show_weights: bool = True,
) -> str:
    """Build a clean text / ASCII representation of a decision tree."""
    if decision_tree.tree_ is None:
        return "Tree is empty (not fitted yet)."

    if feature_names is None:
        names = decision_tree.feature_names_in_
    else:
        names = list(feature_names)

    lines: List[str] = []

    def _recurse(node: Node, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return

        indent = "|   " * depth
        if node.is_leaf:
            imp = abs(node.impurity)
            info = f"class: {node.predicted_class}"
            if show_weights:
                info += f" (samples: {node.samples}, impurity: {imp:.3f})"
            lines.append(f"{indent}|--- {info}")
            return

        feat = (
            names[node.feature_idx]
            if node.feature_idx < len(names)
            else f"feature_{node.feature_idx}"
        )
        thresh = node.threshold if node.threshold is not None else 0.5

        left_cond = f"{feat} <= {thresh:.2f}"
        right_cond = f"{feat} >  {thresh:.2f}"

        lines.append(f"{indent}|--- {left_cond}")
        if node.left:
            _recurse(node.left, depth + 1)

        lines.append(f"{indent}|--- {right_cond}")
        if node.right:
            _recurse(node.right, depth + 1)

    _recurse(decision_tree.tree_, depth=0)
    return "\n".join(lines)


def plot_tree(
    decision_tree: DecisionTreeClassifier,
    max_depth: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    filled: bool = True,
    rounded: bool = True,
    precision: int = 1,
    ax: Optional[Any] = None,
    fontsize: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    title: Optional[str] = "Step 4: Full Decision Tree (All Leaf Nodes Isolated)",
) -> Any:
    """Plot a decision tree with collision-free, ultra-wide layout matching official styling."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plot_tree(). Install it with: pip install matplotlib"
        ) from e

    if decision_tree.tree_ is None:
        raise RuntimeError(
            "This DecisionTreeClassifier instance is not fitted yet. Call 'fit' first."
        )

    root = decision_tree.tree_
    names = (
        decision_tree.feature_names_in_
        if feature_names is None
        else list(feature_names)
    )
    classes = list(decision_tree.classes_) if class_names is None else list(class_names)
    criterion_name = (
        "entropy"
        if decision_tree.criterion in ("entropy", "information_gain")
        else "gini"
    )

    def _is_effective_leaf(node: Node, depth: int) -> bool:
        if node.is_leaf:
            return True
        if max_depth is not None and depth >= max_depth:
            return True
        return False

    def _count_effective_leaves(node: Node, depth: int) -> int:
        if _is_effective_leaf(node, depth):
            return 1
        count = 0
        if node.left:
            count += _count_effective_leaves(node.left, depth + 1)
        if node.right:
            count += _count_effective_leaves(node.right, depth + 1)
        return max(1, count)

    total_leaves = _count_effective_leaves(root, 0)
    actual_depth = decision_tree.get_depth()
    effective_max_depth = (
        min(actual_depth, max_depth) if max_depth is not None else actual_depth
    )

    # Compute collision-free horizontal coordinates
    positions: Dict[Node, Tuple[float, float]] = {}
    leaf_idx = 0

    def _assign_coords(node: Node, depth: int) -> Tuple[float, float]:
        nonlocal leaf_idx
        if _is_effective_leaf(node, depth):
            leaf_idx += 1
            x = (leaf_idx - 0.5) / total_leaves
        else:
            left_x, _ = _assign_coords(node.left, depth + 1)  # type: ignore
            right_x, _ = _assign_coords(node.right, depth + 1)  # type: ignore
            x = (left_x + right_x) / 2.0

        if effective_max_depth > 0:
            y = 0.93 - (depth / effective_max_depth) * 0.86
        else:
            y = 0.5

        positions[node] = (x, y)
        return (x, y)

    _assign_coords(root, 0)

    # Large, spacious canvas size guaranteeing no horizontal box collisions
    if ax is None:
        if figsize is None:
            fig_w = max(38.0, total_leaves * 2.4)
            fig_h = max(20.0, (effective_max_depth + 1) * 3.2)
            figsize = (fig_w, fig_h)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=16, pad=25, fontweight="normal")

    if fontsize is None:
        fontsize = 8.5

    boxstyle = "round,pad=0.35,rounding_size=0.15" if rounded else "square,pad=0.35"

    # Color palette for classes (vibrant solid colors matching official plot)
    cmap = matplotlib.colormaps["tab20"].resampled(max(1, len(classes)))

    # Draw connecting edges with clean lines & arrowheads
    def _draw_edges(node: Node, depth: int) -> None:
        if _is_effective_leaf(node, depth):
            return

        px, py = positions[node]

        for is_left, child in ((True, node.left), (False, node.right)):
            if child is None:
                continue
            cx, cy = positions[child]

            ax.plot([px, cx], [py, cy], color="#0F172A", lw=1.0, zorder=1)

            if depth == 0:
                mx = px * 0.45 + cx * 0.55
                my = py * 0.45 + cy * 0.55 + 0.015
                branch_lbl = "True" if is_left else "False"
                ax.text(
                    mx,
                    my,
                    branch_lbl,
                    fontsize=fontsize + 1.0,
                    color="#0F172A",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=2,
                )

            _draw_edges(child, depth + 1)

    _draw_edges(root, 0)

    def _format_value_list(val_arr: Optional[np.ndarray]) -> str:
        if val_arr is None:
            return ""
        val_list = [int(v) for v in val_arr]
        if len(val_list) > 8:
            chunks = [val_list[i : i + 8] for i in range(0, len(val_list), 8)]
            return "value = [" + ",\n         ".join(str(c)[1:-1] for c in chunks) + "]"
        return f"value = {val_list}"

    # Draw node boxes
    def _draw_nodes(node: Node, depth: int) -> None:
        x, y = positions[node]
        is_leaf_render = _is_effective_leaf(node, depth)

        lines_text = []
        val_str = _format_value_list(node.value)

        if is_leaf_render:
            pred_cls = node.predicted_class
            lines_text.append(f"{criterion_name} = {abs(node.impurity):.{precision}f}")
            lines_text.append(f"samples = {node.samples}")
            if val_str:
                lines_text.append(val_str)
            lines_text.append(f"class = {pred_cls}")

            if filled:
                try:
                    cls_idx = classes.index(pred_cls)
                except ValueError:
                    cls_idx = 0
                fc = cmap(cls_idx % 20)
                ec = "#0F172A"
            else:
                fc = "#FFFFFF"
                ec = "#0F172A"
        else:
            feat = (
                names[node.feature_idx]
                if node.feature_idx < len(names)
                else f"feature_{node.feature_idx}"
            )
            thresh = node.threshold if node.threshold is not None else 0.5
            lines_text.append(f"{feat} <= {thresh:.1f}")
            lines_text.append(f"{criterion_name} = {abs(node.impurity):.{precision}f}")
            lines_text.append(f"samples = {node.samples}")
            if val_str:
                lines_text.append(val_str)
            lines_text.append(f"class = {node.predicted_class}")

            fc = "#FFFFFF"
            ec = "#0F172A"

        node_str = "\n".join(lines_text)
        ax.text(
            x,
            y,
            node_str,
            ha="center",
            va="center",
            fontsize=fontsize,
            bbox=dict(
                boxstyle=boxstyle,
                fc=fc,
                ec=ec,
                lw=0.9,
            ),
            zorder=3,
        )

        if not is_leaf_render:
            if node.left:
                _draw_nodes(node.left, depth + 1)
            if node.right:
                _draw_nodes(node.right, depth + 1)

    _draw_nodes(root, 0)
    plt.tight_layout()
    return ax
