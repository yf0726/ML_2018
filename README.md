# Machine Learning Project 1 on Higgs Boson Competition

`Team`: 4 - 1 = 3 ducks

`Teammates`: Yan Fu, Shengzhao Xia, Wan-Tzu Huang

This project is adapted from the Kaggle 2014 competition [Higgs Boson Machine Learning Challenge] (https://www.kaggle.com/c/Higgs-boson). The competition orgins from the famous Higgs Boson experiment and the competitors need to improve observation of Higgs Boson from background noise using machine learning methods. 

In this project, we are given train set data (including 250000 events, with an ID column, 30 feature columns, a weight column and a label column) and test set data (including 550000 events with an ID column and 30 feature columns). Only Numpy and visualization libraries are allowed. 

To reproduce our result one will need to:

1. Install `python` and `Numpy` on their computers. In order to reproduce perfectly, please make sure that the versions are `python3.6` and `Numpy1.14.3`.
2. Download `train.csv` and `test.csv` from [kaggle](https://www.kaggle.com/c/epfml18-higgs), and save them in `/data` document;
3. Run the python scripy `run.py`.

Below we will introduce files and functions in our repotory.

## Auxiliary modules

* ### `helpers.py`

Contains the given helper functions, including `load_csv_data` (used to load csv file) and `create_csv_submission` (used to create output files in csv format for Kaggle submission).

* ### `tools.py`

  * `DataFrame` The class is used to replace the pandas.DataFrame with Numpy methods, including copy, reset_index, drop, loc and value counts.

And in tools.py we also include functions used for data processing.

  * `standardize`: Standardize data by minus data mean and then divide by data's standard deviation.

  * `log_process`: Take logarithm of positive data to narrow data ranging.

  * `build_polynomial_features`: Build polynomial features by squaring, root-squaring and multiplying bewteen features. 

Besides, in tools.py we include the cross validation functions using `predict_labels`, `build_k_indices`, `cross_validation` and `cv_loop`.

## Regression models
* ### `implementations.py`

Contains 6 implemented model functions, including `least squares GD`, `least squares SGD`, `least squares`, `ridge regression`, `logistic regression` and `reg logistic regression`.

In addition, we also include other function that are needed by the model functions (namely, cost functions `calculate_logi_loss`, `calculate_mse`, `calculate_mae` and `compute_loss` ,sigmoid function `sigmoid` and (stochastic) gradient descent `batch_iter` and `compute_gradient`).

