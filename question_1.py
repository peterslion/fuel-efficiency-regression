"""Question 1: Which of the feature columns has missing values?"""

from dataset import FEATURE_COLUMNS, load_dataset


def main() -> None:
    frame = load_dataset()
    missing_counts = frame[FEATURE_COLUMNS].isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    for column in FEATURE_COLUMNS:
        print(f"{column}: {int(missing_counts[column])} missing")

    if len(columns_with_missing) != 1:
        raise RuntimeError(
            "Expected exactly one feature column with missing values, "
            f"found {list(columns_with_missing.index)}"
        )

    answer = str(columns_with_missing.index[0])
    print(f"ANSWER: {answer}")


if __name__ == "__main__":
    main()
