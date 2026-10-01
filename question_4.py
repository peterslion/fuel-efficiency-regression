"""Question 4: Which regularization strength r has the best validation RMSE?"""

from dataset import extract_y, load_dataset, prepare_X, train_val_test_split
from linear_regression import predict, rmse, train_linear_regression_reg

R_VALUES = [0, 0.01, 0.1, 1, 5, 10, 100]


def main() -> None:
    df_train, df_val, _df_test = train_val_test_split(load_dataset(), seed=42)
    X_train = prepare_X(df_train, fill_value=0.0)
    X_val = prepare_X(df_val, fill_value=0.0)
    y_train = extract_y(df_train)
    y_val = extract_y(df_val)

    scores = []
    for r in R_VALUES:
        w0, w = train_linear_regression_reg(X_train, y_train, r=r)
        score = round(rmse(y_val, predict(X_val, w0, w)), 4)
        scores.append((score, r))
        print(f"r={r}: {score:.4f}")

    # Lower RMSE wins. Ties keep the smallest r because scores are scanned
    # in ascending r and equality does not replace the current best.
    best_score, best_r = min(scores, key=lambda item: (item[0], item[1]))
    print(f"ANSWER: {best_r}")
    print(f"best_rmse: {best_score:.4f}")


if __name__ == "__main__":
    main()
