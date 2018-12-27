# Machine Learning Project 2 :Machine Learning for Science

`Teammates`: Yan Fu, Shengzhao Xia, Runzhe Liu

For this project we joined the lab in Chair of Economics and Management of Innovation. 

In this project, we did:

1. Data collection: web scrap from Web of Science and get over 350,000 paper records;
2. Name disambiguation: Disambiguation of different authors using same name;
3. Classification: Using models (eg. logistic regression, neural network), and converting keywords as vectors using word2vec to predict if one researcher is dismissed or not.

To reproduce our result one will need to:

1. Install libraries like `gensim`,`sklearn` on their computers;
2. Download [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) from [kaggle](https://www.kaggle.com/c/epfml18-higgs), we did not include it in our submission because of its large size.

Below we will introduce files and functions in our repotory.

## Auxiliary modules
放爬虫和数据处理的文件

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
放模型
* ### `implementations.py`

Contains 6 implemented model functions, including `least squares GD`, `least squares SGD`, `least squares`, `ridge regression`, `logistic regression` and `reg logistic regression`.

In addition, we also include other function that are needed by the model functions (namely, cost functions `calculate_logi_loss`, `calculate_mse`, `calculate_mae` and `compute_loss` ,sigmoid function `sigmoid` and (stochastic) gradient descent `batch_iter` and `compute_gradient`).

## Hyper-parameter tuning using Grid Search
