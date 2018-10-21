# -*- coding: utf-8 -*-

import numpy as np

# -----------   helpers            ---------- #
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def standardize(x):
    """Standardize the original data set."""
    x_standardize = np.empty_like(x, dtype='float64')

    mean_x = np.mean(x, axis=0) #x 为一个ndarray或array-like对象
    for colomn in range(x.shape[1]):
        x_standardize[:, colomn] = x[:, colomn] - mean_x[colomn]

    std_x = np.std(x, axis=0)
    for colomn in range(x.shape[1]):
        x_standardize[:, colomn] = x_standardize[:, colomn] / std_x[colomn]

    return x_standardize, mean_x, std_x



# ----------    Cost calculation    ---------- #
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_logi_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred )) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def compute_loss(y, tx, w, method = calculate_mse):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)   # 一维的np.array都视作'列'向量: 从其尺寸(2000,)也能看出，2000行，列向量
    return method(e)
    # return calculate_mae(e)

# ----------    Gradient descent    ---------- #
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad

def gradient_descent(y, tx, initial_w, max_iters, gamma, method = calculate_mae):   # max_iters为最大迭代次数
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]    # 提前加上[]，变为list，[[w1，w2]]
    losses = []
    w = initial_w
    for n_iter in range(max_iters): # range返回迭代器
        # TODO: compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w, method)
        # TODO: update w by gradient
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1])) # {}中可以指定任意关键字
    return losses, ws

# ----------    Stochastic descent    ---------- #
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, method = calculate_mae):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    ws = [initial_w] # Store the w in the process
    losses = []

    for i in range(max_iters):
        for y_batch, x_batch in batch_iter(y, tx, batch_size, 1, True): # Since num_batches=1, the loop run only once
            w = w - gamma * compute_gradient(y_batch, x_batch, w)
            loss = compute_loss(y, tx, w, method) # Every new w match a loss, ingore the original loss
            losses.append(loss)
            ws.append(w)
            print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(\
                bi=i, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Input: Two iterables
    Output: An iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Note: Data can be randomly shuffled.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
# 从 w 中随机选出 batch_size 个方向作为 L(w) 的下降梯度，此处抽离这些方向对应的  (y,tx) 用于计算梯度
    data_size = y.shape[0]  # len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))   # permutation：排列
        shuffled_y = y[shuffle_indices] # Gets the view corresponding to the indices. NB : iterable of arrays as indexing
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        # 这里是考虑了可能抽取多个(num_batches个）
        start_index = batch_num * batch_size     # 洗牌数据后，连续抽num_batches个，相当于随机抽取
        end_index = min((batch_num + 1) * batch_size, data_size)   # 防止越界
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

# ----------    Least squares    ---------- #
def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def least_squares_regression(y, x):
    """linear regression demo."""
    weights = least_squares(y, x)
    rmse = np.sqrt(2 * compute_loss(y, x, weights))
    print("rmse={loss}".format(loss=rmse))

# ----------    Ridge Regression    ---------- #
def ridge_regression_solve(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def ridge_regression(y, x):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-15, 5, 50)
    rmse_tr = []
    for ind, lambda_ in enumerate(lambdas):
        # ridge regression
        weight = ridge_regression_solve(y, x, lambda_)
        rmse_tr.append(np.sqrt(2 * compute_loss(y, x, weight)))
        print("lambda={l:.3f}, Training RMSE={tr:.3f}".format(l=lambda_, tr=rmse_tr[ind]))


# ----------    Logistic regression    ---------- #
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_logi_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + 1e-12)) + (1 - y).T.dot(np.log(1 - pred + 1e-12))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """

    loss = calculate_logi_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

def logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma):
    # init parameters
    losses = []
    w = initial_w
    threshold = 1e-8
    y = y.reshape(len(y),1)
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return weight, losses[-1]

# ----------    Regularized logistic regression    ---------- #
def penalized_logistic_regression(y, tx, w, lambda_):
    y = y.reshape(len(y),1)
    loss = calculate_logi_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = tx.T.dot(1.0 / (1 + np.exp(-tx.dot(w))) - y) + 2 * lambda_ * w
    return loss, gradient

def regularized_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    # Set default weight
    weight = initial_w
    losses = []
    y = y.reshape(len(y),1)
    threshold = 1e-8

    for i in range(max_iters):

        # get loss and update w.
        loss, gradient = penalized_logistic_regression(y, tx, weight, lambda_)
        weight = weight - gamma * gradient
        losses.append(loss)

        # log info
        #if iter % 100 == 0:
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        # termination condition
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return weight, losses[-1]