# Beures Housing — Buenos Aires Apartment Price Prediction

Predicting apartment prices in Buenos Aires, Argentina, using linear and regularized regression models (Linear Regression, Ridge, Lasso). The project walks through the full applied ML workflow: data cleaning, feature engineering, exploratory analysis, model training, hyperparameter tuning via cross-validation, and model selection/evaluation.

## Overview

Raw real-estate listings for Buenos Aires are messy: they mix multiple cities/regions, contain outliers, encode latitude/longitude and place names as strings, and have columns with heavy missingness. This project builds a repeatable cleaning pipeline, then trains and compares several regression models to predict `price_aprox_usd` for apartments in Capital Federal (the autonomous city of Buenos Aires).

Key steps covered:
- Merging multiple raw CSV files with `glob`
- Filtering to Capital Federal apartments under $400K
- Removing outliers in `surface_covered_in_m2` using a quantile-based approach
- Feature engineering (extracting `lat`/`lon` from a combined `lat-lon` string, extracting `neighborhood` from `place_with_parent_names`)
- Missing-value analysis (including visualization with `missingno`)
- Multicollinearity analysis via correlation heatmaps
- Baseline, Linear Regression, Ridge, and Lasso models built with `scikit-learn` pipelines
- Bias-variance tradeoff exploration across a range of regularization strengths (alpha)
- Automated hyperparameter selection with `RidgeCV` and `LassoCV`
- Coefficient interpretation and feature importance (Lasso's built-in feature selection)
- Residual analysis and predicted-vs-actual diagnostic plots
- A structured final model recommendation based on test-set RMSE

## Repository Structure

```
beures_housing/
├── buenos_aires_real_estate.ipynb        # Data cleaning, EDA, and a baseline/simple Linear Regression model
├── Beures_Housing_model.ipynb            # Ridge & Lasso pipelines, bias-variance tradeoff, coefficient analysis
├── Beures_housing_model_selection.ipynb  # Baseline vs. Linear/Ridge/LassoCV comparison and final model recommendation
├── LICENSE                               # MIT License
└── README.md
```

## Dataset

The notebooks expect one or more CSV files named following the pattern:

```
buenos-aires-real-estate-*.csv
```

Each file is a listing export containing columns such as `place_with_parent_names`, `lat-lon`, `surface_covered_in_m2`, `price_aprox_usd`, `rooms`, `floor`, and `expenses`, among others.

> **Note:** The dataset itself is not included in this repository. The notebooks were originally developed on Google Colab and read data from Google Drive (`/content/drive/MyDrive/...`). To run locally, place your CSV file(s) in a local folder and update the `pattern` variable accordingly, and remove/skip the `google.colab` drive-mount cells.

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab (or Google Colab)

### Installation

Clone the repository:

```bash
git clone https://github.com/dositadi/beures_housing.git
cd beures_housing
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn missingno category_encoders jupyter
```

### Running the Notebooks

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open the notebooks in order:
   - `buenos_aires_real_estate.ipynb` — start here for data cleaning and the baseline model
   - `Beures_Housing_model.ipynb` — Ridge/Lasso pipelines and regularization analysis
   - `Beures_housing_model_selection.ipynb` — final model comparison and recommendation
3. If running outside Colab, remove the `drive.mount(...)` cell and point the `pattern` variable to your local data directory.

Each notebook also includes an "Open in Colab" badge for running directly in Google Colab without local setup (Colab users will need their own copy of the dataset in Google Drive).

## Methodology

| Model | Type | Regularization | Hyperparameter Selection |
|---|---|---|---|
| Baseline | Mean predictor | None | None |
| Linear Regression | OLS | None | None (closed-form) |
| Ridge | L2 regularized | α · Σβⱼ² | Automatic via `RidgeCV` |
| Lasso | L1 regularized | α · Σ\|βⱼ\| | Automatic via `LassoCV` |

Models are built as `scikit-learn` pipelines combining:
- `category_encoders.OneHotEncoder` for categorical features (e.g., neighborhood)
- `StandardScaler` for feature scaling
- The regression estimator itself

Evaluation is based primarily on **RMSE** (Root Mean Squared Error) on held-out test data, with **MAE** and **R²** also reported for the baseline linear model. Diagnostics include predicted-vs-actual plots and residual plots to check for systematic bias.

## Results

Final model selection and a structured recommendation (which model performs best on test RMSE, whether the difference is practically meaningful, and Lasso's feature-selection benefits) are documented at the end of `Beures_housing_model_selection.ipynb`.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project builds on real-estate data and exercises adapted from the WorldQuant University Applied Data Science curriculum.
