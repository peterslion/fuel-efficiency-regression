"""Question 2: Median (50% percentile) of horsepower."""

import numpy as np

from dataset import load_dataset


def main() -> None:
    horsepower = load_dataset()["horsepower"]
    pandas_median = float(horsepower.median())
    numpy_median = float(np.nanpercentile(horsepower.to_numpy(), 50))
    if pandas_median != numpy_median:
        raise RuntimeError(
            f"Median implementations disagree: {pandas_median} vs {numpy_median}"
        )

    print(f"non_null: {int(horsepower.notna().sum())}")
    print(f"missing: {int(horsepower.isna().sum())}")
    print(f"median: {pandas_median:.1f}")
    # The 50% percentile is an integer horsepower value.
    print(f"ANSWER: {pandas_median:.0f}")


if __name__ == "__main__":
    main()
