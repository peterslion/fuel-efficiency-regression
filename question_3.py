"""Question 3: Compare filling missing horsepower with 0 vs the training mean."""

from dataset import extract_y, load_dataset, prepare_X, train_val_test_split
from linear_regression import predict, rmse, train_linear_regression


def validation_rmse(df_train, df_val, fill_value: float) -> float:
    X_train = prepare_X(df_train, fill_value)
    X_val = prepare_X(df_val, fill_value)
    y_train = extract_y(df_train)
    y_val = extract_y(df_val)

    w0, w = train_linear_regression(X_train, y_train)
    return rmse(y_val, predict(X_val, w0, w))


def main() -> None:
    df_train, df_val, _df_test = train_val_test_split(load_dataset(), seed=42)
    # The mean is computed on the training fold only, and NaNs are skipped.
    horsepower_mean = float(df_train["horsepower"].mean())

    rmse_zero = validation_rmse(df_train, df_val, fill_value=0.0)
    rmse_mean = validation_rmse(df_train, df_val, fill_value=horsepower_mean)

    rmse_zero_rounded = round(rmse_zero, 3)
    rmse_mean_rounded = round(rmse_mean, 3)
    better = "mean" if rmse_mean_rounded < rmse_zero_rounded else "0"

    print(f"train_rows: {len(df_train)}")
    print(f"val_rows: {len(df_val)}")
    print(f"training_horsepower_mean: {horsepower_mean:.6f}")
    print(f"rmse_fill_0: {rmse_zero_rounded:.3f}")
    print(f"rmse_fill_mean: {rmse_mean_rounded:.3f}")
    print(f"ANSWER: {better}")


if __name__ == "__main__":
    main()
