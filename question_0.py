"""Question 0: Does fuel_efficiency_mpg have a long tail?"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import TARGET, load_dataset

FIGURE_PATH = Path(__file__).resolve().parent / "figures" / "q0_fuel_efficiency_mpg.png"


def has_long_tail(skewness: float) -> bool:
    """A long tail here means strong right skew, the shape a log transform fixes.

    |skew| below 0.5 is approximately symmetric. The Zoomcamp long-tail
    examples (price, MSRP) sit well above 1.
    """
    return skewness > 1.0


def main() -> None:
    fuel_efficiency = load_dataset()[TARGET]
    mean = float(fuel_efficiency.mean())
    median = float(fuel_efficiency.median())
    skewness = float(fuel_efficiency.skew())
    excess_kurtosis = float(fuel_efficiency.kurtosis())
    long_tail = has_long_tail(skewness)

    figure_path = FIGURE_PATH
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(
        fuel_efficiency,
        bins=40,
        color="#3d5a80",
        edgecolor="white",
        linewidth=0.4,
    )
    axis.axvline(mean, color="#e07a5f", linestyle="--", linewidth=1.5, label=f"mean {mean:.2f}")
    axis.axvline(median, color="#81b29a", linestyle="-", linewidth=1.5, label=f"median {median:.2f}")
    axis.set_xlabel("fuel_efficiency_mpg")
    axis.set_ylabel("Number of cars")
    axis.set_title("Fuel efficiency is nearly symmetric")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)

    print(f"count: {len(fuel_efficiency)}")
    print(f"min: {float(fuel_efficiency.min()):.1f}")
    print(f"max: {float(fuel_efficiency.max()):.1f}")
    print(f"mean: {mean:.4f}")
    print(f"median: {median:.4f}")
    print(f"skewness: {skewness:.4f}")
    print(f"excess_kurtosis: {excess_kurtosis:.4f}")
    print(f"max/median: {float(fuel_efficiency.max()) / median:.3f}")
    print(f"figure: {figure_path}")
    print(f"ANSWER: {'Yes' if long_tail else 'No'}")


if __name__ == "__main__":
    main()
