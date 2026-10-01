# Car fuel efficiency regression

Homework that fits a linear regression model to predict `fuel_efficiency_mpg`.

The model uses only:

- `engine_displacement`
- `horsepower`
- `vehicle_weight`
- `model_year`
- `fuel_efficiency_mpg` (target)

The table is the 2026 Machine Learning Zoomcamp car fuel-efficiency file, stored at `data/car_fuel_efficiency_2026.csv`. Source:

```text
https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

## Run

```bash
python3 -m pip install -r requirements.txt
python3 question_0.py
python3 question_1.py
python3 question_2.py
```

Each `question_*.py` script prints its answer on the `ANSWER:` line.

## Answers

| Question | Answer |
| --- | --- |
| 0. Does `fuel_efficiency_mpg` have a long tail? | No |
| 1. Which column has missing values? | `horsepower` (877 missing) |
| 2. Median of `horsepower` | 254 |
