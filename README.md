# Tehran House Price Prediction

An end-to-end regression notebook exploring residential property prices in Tehran. The project uses listing attributes such as area, room count, building amenities, and neighborhood to compare several machine-learning approaches and evaluate how well they estimate the listed price.

## Project highlights

- Exploratory data analysis and visualizations of Tehran housing listings
- Feature preparation for numeric, Boolean, and neighborhood data
- Comparison of eight regression models, from linear baselines to tree ensembles
- Hyperparameter search with cross-validation and a held-out test set
- Evaluation using **R²** and **root mean squared error (RMSE)**

## Results

The notebook's recorded 80/20 train-test split used `random_state=0`. The table below summarizes the saved test-set results; RMSE is shown in the raw units of the `Price` column.

| Model | Test R² | Test RMSE |
|---|---:|---:|
| Linear Regression | 53.35% | 6.36B |
| Ridge | 53.36% | 6.36B |
| Lasso | 53.35% | 6.36B |
| Elastic Net | 55.36% | 6.22B |
| Decision Tree | 71.37% | 4.98B |
| **Random Forest** | **77.55%** | **4.41B** |
| K-Nearest Neighbors | 63.99% | 5.59B |
| XGBoost | 76.72% | 4.49B |

Random Forest achieved the highest recorded test R² in this experiment. These are notebook results, not a guarantee of performance on new listings. They should be treated as a reference: the saved run used 3,456 rows, while the CSV currently included in this repository has 3,479 rows, including 23 missing neighborhood values.

## Repository contents

| File | Description |
|---|---|
| [`tehran-divar-ir-house-price-prediction.ipynb`](./tehran-divar-ir-house-price-prediction.ipynb) | Data exploration, preprocessing, model training, and evaluation |
| [`tehranhouses.csv`](./tehranhouses.csv) | Housing listings dataset (3,479 rows, 8 columns) |

## Getting started

1. Clone the repository and move into the project directory.
2. Create and activate a Python environment, then install the notebook dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn xgboost
   ```

3. Open the notebook:

   ```bash
   jupyter notebook tehran-divar-ir-house-price-prediction.ipynb
   ```

4. In the data-loading cell, point `pd.read_csv(...)` to the CSV in this repository:

   ```python
   df = pd.read_csv("tehranhouses.csv")
   ```

5. Run the notebook cells from top to bottom.

The notebook's original data-loading path refers to a Kaggle input directory, so update it as shown above when running locally. The provided CSV has 23 missing values in `Address`; the notebook uses one-hot encoding for this feature, and the resulting metrics can differ from the saved results.

## Data and workflow

The CSV contains 3,479 listings and the following fields:

| Column | Description |
|---|---|
| `Area` | Property area |
| `Room` | Number of rooms |
| `Parking` | Parking availability |
| `Warehouse` | Storage availability |
| `Elevator` | Elevator availability |
| `Address` | Neighborhood/address label |
| `Price` | Regression target |
| `Price(USD)` | Additional price field, excluded from model features |

The notebook explores the data and price distributions, converts amenity flags to numeric features, one-hot encodes `Address`, and compares eight regressors. Hyperparameters for selected models are searched with `GridSearchCV`; final scores are reported on a held-out test set.

## Scope and limitations

- This is a notebook-based experimental project; it does not include a deployed prediction API or user interface.
- The target is the dataset's `Price` field. Its currency and unit are not formally documented here, so RMSE values are reported in raw target units.
- Twenty-three records in the included CSV have no `Address`. Consider adding an explicit missing-value strategy and evaluating its effect.
- The saved metrics and the included CSV do not represent the exact same row count; rerun the analysis on a consistently prepared dataset before comparing or presenting updated scores.

## Potential next steps

- Add a reproducible preprocessing pipeline and data validation
- Tune feature engineering and missing-value handling
- Use cross-validation for more robust model comparison
- Track model artifacts and document the target's currency and units
