# -*- coding: utf-8 -*-

import numpy as np
            
# ----------    Cost calculation    ---------- #
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w, method = calculate_mse):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)   # 一维的np.array都视作'列'向量: 从其尺寸(2000,)也能看出，2000行，列向量
    return calculate_mse(e)
    # return calculate_mae(e)

# ----------    Gradient descent    ---------- #
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad

def gradient_descent(y, tx, initial_w, max_iters, gamma, method = calculate_mse):   # max_iters为最大迭代次数
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
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, method = calculate_mse):
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
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    return loss,w
# ----------    Logistic regression    ---------- #
def sigmoid(t):
    """apply sigmoid function on t."""
    logistic_value = np.exp(t) / (1 + np.exp(t))
    logistic_value[np.isnan(logistic_value)] =1

    return logistic_value

def calculate_logi_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + 1e-12)) + (1 - y).T.dot(np.log(1 - pred + 1e-12))  # avoid 0
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
    y = (y + 1) / 2     # make -1,1 values to 0,1
    loss = calculate_logi_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

def logistic_regression_GD(y, tx, max_iters):
    # init parameters
    losses = []
    w = np.zeros((tx.shape[1], 1))
    threshold = 1e-8
    y = y.reshape(len(y),1)
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        eignvalue,_ = np.linalg.eig(tx.T.dot(tx))
        gamma = 1/(0.5*eignvalue.max())
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        # if iter % 10 == 0:
            # print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return losses[-1],w

# ----------    Regularized logistic regression    ---------- #
def penalized_logistic_regression(y, tx, w, lambda_):
    loss = calculate_logi_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = tx.T.dot(1.0 / (1 + np.exp(-tx.dot(w))) - y) + 2 * lambda_ * w
    return loss, gradient

def regularized_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold):

    # Set default weight
    y = (y + 1) / 2
    weight = initial_w
    losses = []

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
