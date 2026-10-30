import os
import sys
import csv
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from cmlmath import DecisionTreeClassifier, Node, export_text, plot_tree


class Character(object):
    """Character dataset manager and Guess Who player."""

    dataset_csv: os.PathLike

    dataset: Optional[csv.DictReader]

    def __init__(
        self, dataset_csv: Optional[Union[str, os.PathLike]] = None, *args, **kwargs
    ) -> None:
        if dataset_csv is None:
            # Default to characters.csv in the same directory as this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.dataset_csv = os.path.join(base_dir, "characters.csv")
        else:
            self.dataset_csv = os.path.abspath(dataset_csv)

        self.rows: List[Dict[str, str]] = []
        self.feature_names: List[str] = []
        self.names: List[str] = []

        self.load()

    def load(self) -> None:
        """Load character records from the CSV file."""
        if not os.path.exists(self.dataset_csv):
            raise FileNotFoundError(f"Characters CSV not found at: {self.dataset_csv}")

        with open(self.dataset_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.dataset = reader
            fieldnames = reader.fieldnames or []
            self.feature_names = [f for f in fieldnames if f.lower() != "name"]
            self.rows = list(reader)
            self.names = [r["name"] for r in self.rows]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert the loaded dataset into (X, y, feature_names) for DecisionTreeClassifier."""
        X_list = []
        y_list = []

        for row in self.rows:
            y_list.append(row["name"])
            feature_vals = []
            for feat in self.feature_names:
                raw_val = row.get(feat, "").strip().upper()
                feature_vals.append(1 if raw_val in ("TRUE", "1", "YES", "T") else 0)
            X_list.append(feature_vals)

        X = np.ascontiguousarray(X_list, dtype=np.int32)
        y = np.array(y_list)
        return X, y, self.feature_names

    def get_character_traits(self, name: str) -> Optional[Dict[str, bool]]:
        """Get the dictionary of traits for a given character by name."""
        for row in self.rows:
            if row["name"].upper() == name.upper():
                return {
                    f: (row[f].strip().upper() in ("TRUE", "1", "YES", "T"))
                    for f in self.feature_names
                }
        return None

    def play_akinator(
        self,
        clf: DecisionTreeClassifier,
        target_character: Optional[str] = None,
    ) -> str:
        """Simulate or interactively play the Guess Who game using the decision tree."""
        if clf.tree_ is None:
            raise ValueError("Classifier must be fitted first.")

        if target_character:
            traits = self.get_character_traits(target_character)
            if traits is None:
                raise ValueError(f"Unknown character '{target_character}'.")
            print(
                f"\n--- [Akinator] Target character to identify: {target_character.upper()} ---"
            )
        else:
            traits = None
            print("\n--- [Akinator] Think of a character from the dataset! ---")

        curr: Node = clf.tree_
        question_count = 0

        while not curr.is_leaf:
            question_count += 1
            feat = curr.feature_name or f"feature_{curr.feature_idx}"

            if feat.startswith("Is"):
                trait_question = f"Is the character {feat[2:]}?"
            elif feat.startswith("Has"):
                trait_question = f"Does the character have {feat[3:]}?"
            else:
                trait_question = f"Does the character have '{feat}'?"

            if traits is not None:
                has_trait = traits[feat]
                answer = "YES" if has_trait else "NO"
                print(f"Q{question_count}: {trait_question:45} Answer: {answer}")
            else:
                resp = (
                    input(f"Q{question_count}: {trait_question} (y/n): ")
                    .strip()
                    .lower()
                )
                has_trait = resp.startswith("y")

            if has_trait:
                curr = curr.right
            else:
                curr = curr.left

        guessed_character = str(curr.predicted_class)
        status = (
            "CORRECT!"
            if (
                target_character is None
                or guessed_character.upper() == target_character.upper()
            )
            else "FAILED"
        )
        print(
            f"Result: The character is {guessed_character}! ({status} - Solved in {question_count} questions)\n"
        )
        return guessed_character


def main() -> None:
    print("=" * 65)
    print("CoreMLMath DecisionTreeClassifier Demo: Guess Who / Characters")
    print("=" * 65)

    char_data = Character()
    X, y, feature_names = char_data.to_numpy()

    print(f"\nLoaded dataset with {X.shape[0]} characters and {X.shape[1]} features.")
    print(f"Characters: {', '.join(char_data.names)}")

    clf = DecisionTreeClassifier(
        criterion="entropy",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    print("\nFitting DecisionTreeClassifier using multithreaded CoreMLMath backend...")
    clf.fit(X, y, feature_names=feature_names)

    score = clf.score(X, y)
    print(f"Training Accuracy: {score * 100:.1f}%")
    print(f"Tree Depth: {clf.get_depth()}")
    print(f"Total Leaves: {clf.get_n_leaves()}")

    print("\nFeature Importances (Impurity Decrease Contribution):")
    sorted_features = sorted(
        zip(clf.feature_names_in_, clf.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    for feat, imp in sorted_features:
        if imp > 0.0:
            print(f"  - {feat:25} {imp:.4f} ({imp * 100:.1f}%)")

    print("\nLearned Decision Tree Structure:")
    print("-" * 65)
    clf.print_tree()
    print("-" * 65)

    print("\nSimulating Guess Who game for selected characters:")
    for test_char in ["ZOE", "ERIC", "ALEX", "CHANTAL"]:
        char_data.play_akinator(clf, target_character=test_char)

    plot_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "decision_tree.png"
    )
    print(f"\nPlotting decision tree with matplotlib to '{plot_path}'...")
    try:
        clf.plot_tree(save_path=plot_path, dpi=180)
        print(f"Successfully generated and saved tree plot to: {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
