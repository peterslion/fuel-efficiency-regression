"""Question 5: Standard deviation of validation RMSE across split seeds."""

import numpy as np

from dataset import extract_y, load_dataset, prepare_X, train_val_test_split
from linear_regression import predict, rmse, train_linear_regression

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def validation_rmse(seed: int) -> float:
    df_train, df_val, _df_test = train_val_test_split(load_dataset(), seed=seed)
    X_train = prepare_X(df_train, fill_value=0.0)
    X_val = prepare_X(df_val, fill_value=0.0)
    w0, w = train_linear_regression(X_train, extract_y(df_train))
    return rmse(extract_y(df_val), predict(X_val, w0, w))


def main() -> None:
    scores = np.array([validation_rmse(seed) for seed in SEEDS], dtype=float)
    for seed, score in zip(SEEDS, scores):
        print(f"seed={seed}: {score:.6f}")

    # np.std uses the population divisor (ddof=0), as the homework requests.
    std = float(np.std(scores))
    print(f"std: {std:.6f}")
    print(f"ANSWER: {round(std, 3):.3f}")


if __name__ == "__main__":
    main()
