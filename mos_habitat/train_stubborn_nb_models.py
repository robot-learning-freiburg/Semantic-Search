import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB


def form_dataset(paths):
    X = []
    Y = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        goal_feats = data.item()["goal"]
        non_goal_feats = data.item()["non_goal"]
        if np.any(goal_feats):
            X.append(goal_feats)
            Y.append(np.ones(len(goal_feats)))
        if np.any(non_goal_feats):
            X.append(non_goal_feats)
            Y.append(np.zeros(len(non_goal_feats)))
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y


def train_and_test(path, obj_classes, split_ratio, save_path):
    models = {}
    acc = []
    for obj_class in obj_classes:
        dataitems = list(path.glob(f"{obj_class}*.npy"))
        # assumes data is shuffled by scenes already
        split_idx = int(split_ratio * len(dataitems))
        train_paths = dataitems[:split_idx]
        test_paths = dataitems[split_idx:]

        X_train, Y_train = form_dataset(train_paths)
        X_test, Y_test = form_dataset(test_paths)
        gnb = GaussianNB()
        gnb = gnb.fit(X_train, Y_train)
        y_pred = gnb.predict(X_test)

        print(
            "Number of mislabeled points out of a total %d points : %d"
            % (X_test.shape[0], (Y_test != y_pred).sum())
        )
        # Accuracy of gt label 0
        print(
            "Accuracy of gt label 0: ",
            (Y_test[Y_test == 0] == y_pred[Y_test == 0]).sum()
            / len(Y_test[Y_test == 0]),
        )
        # Accuracy of gt label 1
        print(
            "Accuracy of gt label 1: ",
            (Y_test[Y_test == 1] == y_pred[Y_test == 1]).sum()
            / len(Y_test[Y_test == 1]),
        )
        print(gnb.class_count_)
        print("Accuracy", np.mean(Y_test == y_pred))
        acc.append(np.mean(Y_test == y_pred))
        models[obj_class] = gnb
    print("Overall Accuracy", np.mean(acc))
    pickle.dump(models, open(save_path, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data directory")
    parser.add_argument(
        "--split_ratio", type=float, default=0.95, help="Train-test split ratio"
    )
    parser.add_argument("--save_path", type=str, help="Path to save the trained models")
    args = parser.parse_args()

    path = Path(args.data_path)
    obj_classes = [3, 4, 6, 7, 8]
    split_ratio = args.split_ratio
    save_path = args.save_path

    train_and_test(path, obj_classes, split_ratio, save_path)
