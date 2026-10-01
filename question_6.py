"""Question 6: Test RMSE after training on train+validation with r=0.001."""

import pandas as pd

from dataset import extract_y, load_dataset, prepare_X, train_val_test_split
from linear_regression import predict, rmse, train_linear_regression_reg


def main() -> None:
    df_train, df_val, df_test = train_val_test_split(load_dataset(), seed=9)
    df_full_train = pd.concat([df_train, df_val])

    X_full_train = prepare_X(df_full_train, fill_value=0.0)
    X_test = prepare_X(df_test, fill_value=0.0)
    w0, w = train_linear_regression_reg(
        X_full_train,
        extract_y(df_full_train),
        r=0.001,
    )
    score = rmse(extract_y(df_test), predict(X_test, w0, w))

    print(f"full_train_rows: {len(df_full_train)}")
    print(f"test_rows: {len(df_test)}")
    print(f"rounded_3: {round(score, 3):.3f}")
    print(f"ANSWER: {score:.6f}")


if __name__ == "__main__":
    main()
