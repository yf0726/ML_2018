# ML_2018

`Team`: 4 - 1 = 3 ducks

`Teammates`: Yan Fu, Shengzhao Xia, Wan-Tzu Huang

This project is adapted from the Kaggle 2014 competition [Higgs Boson Machine Learning Challenge] (https://www.kaggle.com/c/Higgs-boson). The competition orgins from the famous Higgs Boson experiment and the competitors need to improve observation of Higgs Boson from background noise using machine learning methods. 

In this project, we are given train set data (including 250000 events, with an ID column, 30 feature columns, a weight column and a label column) and test set data (including 550000 events with an ID column and 30 feature columns). Only Numpy and visualization libraries are allowed. 

To reproduce our result one will need to:

1. Install `Numpy` on their computers;
2. Download `train.csv` and `test.csv` from [kaggle](https://www.kaggle.com/c/epfml18-higgs), and save them in `/data` document;
3. Run the python scripy `run.py`.

Below we will introduce files and functions in our repotory.

## Auxiliary modules

* ### `implementations.py`

Contains 6 implemented model functions, including `least squares GD`, `least squares SGD`, `least squares`, `ridge regression`, `logistic regression` and `reg logistic regression`.

In addition, we also include other function that are needed by the model functions (namely, cost functions `calculate_logi_loss`, `calculate_mse`, `calculate_mae` and `compute_loss` ,sigmoid function `sigmoid` and (stochastic) gradient descent `batch_iter` and `compute_gradient`).

* ### `helpers.py`

Contains the given helper functions, including `load_csv_data` (used to load csv file) and `create_csv_submission` (used to create output files in csv format for Kaggle submission).

* ### `tools.py`

`dataframe` The class is used to replace the pandas.DataFrame with Numpy methods, including 

data processing

`log_process`

`standardize`

`build_polynomial_features`

cross validation
`predict_labels`
`build_k_indices`
`cross_validation`
`cv_loop`

## Regression models

## Hyper-parameter tuning using Grid Search
