"""
tree_model.py - reads a trained LightGBM model file (.txt) and runs its decision trees in plain Python.

The website uses this instead of the LightGBM library, so AgriGate runs on any computer without
installing LightGBM (which needs extra system libraries on some machines, such as libomp on a Mac).
The predictions are the same as LightGBM's: this code follows the same trees, question by question.

    model = TreeModel("models/lightgbm_grown_tuned.txt")
    chances = model.predict(rows)      # rows: one list of feature values per row, in model.feature_names order
"""
import json
import math


class TreeModel:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")

        header = {}
        i = 0
        while i < len(lines) and not lines[i].startswith("Tree="):
            if "=" in lines[i]:
                k, v = lines[i].split("=", 1)
                header[k] = v
            i += 1
        self.feature_names = header["feature_names"].split(" ")
        objective = header.get("objective", "").split(" ")
        self.is_binary = objective[0] == "binary"
        self.sigmoid = 1.0
        for part in objective[1:]:
            if part.startswith("sigmoid:"):
                self.sigmoid = float(part.split(":", 1)[1])

        # one dict of raw fields per tree, until "end of trees"
        self.trees = []
        cur = None
        while i < len(lines) and lines[i] != "end of trees":
            line = lines[i]
            if line.startswith("Tree="):
                cur = {}
                self.trees.append(cur)
            elif cur is not None and "=" in line:
                k, v = line.split("=", 1)
                cur[k] = v
            i += 1
        self.trees = [self._build(t) for t in self.trees]

        # category names LightGBM saw for each categorical feature (in the order those features appear)
        self.pandas_categorical = []
        for line in reversed(lines):
            if line.startswith("pandas_categorical:"):
                self.pandas_categorical = json.loads(line[len("pandas_categorical:"):]) or []
                break

    @staticmethod
    def _build(t):
        leaf_value = [float(x) for x in t["leaf_value"].split()]
        if int(t["num_leaves"]) <= 1:                       # a tree that is a single leaf
            return {"single": leaf_value[0]}
        return {
            "feature": [int(x) for x in t["split_feature"].split()],
            "threshold": [float(x) for x in t["threshold"].split()],
            "decision": [int(x) for x in t["decision_type"].split()],
            "left": [int(x) for x in t["left_child"].split()],
            "right": [int(x) for x in t["right_child"].split()],
            "leaf": leaf_value,
            "cat_boundaries": [int(x) for x in t.get("cat_boundaries", "").split()],
            "cat_threshold": [int(x) for x in t.get("cat_threshold", "").split()],
        }

    @staticmethod
    def _leaf(tree, row):
        """Follow one tree's yes/no questions for one row and return the score of the leaf it lands on."""
        if "single" in tree:
            return tree["single"]
        feature, threshold, decision = tree["feature"], tree["threshold"], tree["decision"]
        left, right = tree["left"], tree["right"]
        node = 0
        while node >= 0:
            v = row[feature[node]]
            d = decision[node]
            if d & 1:                                       # categorical question: "is the category in this set?"
                go_left = False
                if v == v and v >= 0:                       # not missing
                    cat = int(v)
                    idx = int(threshold[node])
                    start, end = tree["cat_boundaries"][idx], tree["cat_boundaries"][idx + 1]
                    word = cat // 32
                    if word < end - start:
                        go_left = (tree["cat_threshold"][start + word] >> (cat % 32)) & 1 == 1
                node = left[node] if go_left else right[node]
            else:                                           # number question: "is the value <= threshold?"
                missing_type = (d >> 2) & 3                 # 0 none, 1 zero, 2 NaN
                is_nan = v != v
                if is_nan and missing_type != 2:
                    v, is_nan = 0.0, False
                if (missing_type == 1 and -1e-35 <= v <= 1e-35) or (missing_type == 2 and is_nan):
                    node = left[node] if d & 2 else right[node]
                else:
                    node = left[node] if v <= threshold[node] else right[node]
        return tree["leaf"][~node]

    def predict_raw(self, rows, num_trees=None):
        trees = self.trees if num_trees is None else self.trees[:num_trees]
        leaf = self._leaf
        return [sum(leaf(t, row) for t in trees) for row in rows]

    def predict(self, rows, num_trees=None):
        """Chance (0-1) for a yes/no model, or the predicted number for a regression model."""
        raw = self.predict_raw(rows, num_trees)
        if not self.is_binary:
            return raw
        s = self.sigmoid
        return [1.0 / (1.0 + math.exp(-s * r)) for r in raw]
