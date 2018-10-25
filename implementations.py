# -*- coding: utf-8 -*-

import numpy as np
import csv

# ----------    Simple DataFrame class ---------- #
class DataFrame:
    def __init__(self, values, index, labels, father_pointer=1):
        self.values = values.copy() \
            if type(values) == type(np.array([])) else np.array(values)
        self.index = index          # still use list to store number index
        self.labels = labels        # still use list to store string labels
        self.father_pointer = self \
            if father_pointer==1 else father_pointer     # refer to the origin DataFrame

    def __getitem__(self,mark):     # []operation, get only one column or row
        if type(mark) == type([]):  # if mark is a list
            if type(mark[0]) == type('string'):
                column = []
                for label in mark:
                    column = column + [i for i, place in enumerate(self.labels) if place == label]    # find the index of the label
                return self.values[:, column].squeeze()
            elif type(mark[0]) == type(0) or type(mark[0]) == type(0.0):# pick a row using number label
                row = []
                for label in mark:
                    row = row + [i for i, place in enumerate(self.index) if place == label]
                return self.values[row, :].squeeze()
        # if mark is not a list
        elif type(mark) == type('string'):                      # pick a column using string label
                column = [i for i, place in enumerate(self.labels) if place == mark]    # find the index of the label
                column = np.array(column)[0]                    # for only 1 line, other wise don't need the [0]
                return self.values[:, column].squeeze()
        elif type(mark) == type(0) or type(mark) == type(0.0):# pick a row using number label
            row = [i for i, place in enumerate(self.index) if place == mark]
            row =np.array(row)[0]
            return self.values[row, :].squeeze()

    def __setitem__(self, key, value):  # assignment overloaded function: a[x] = b
        column = [i for i, place in enumerate(self.labels) if place == key]
        column = np.array(column)[0]       # not need [0], because it is an assignment
        self.values[:, column] = value  # change the value of itself
        temp = np.array(self.index)
        temp = temp - 100000
        self.father_pointer.values[temp, column] = value # change the values in father

    # public methods
    def copy(self):
        values = self.values.copy()
        index = self.index
        labels = self.labels
        return DataFrame(values, index, labels)

    def reset_index(self):     # reset the index as [1,2,3...]
        self.index = range(self.values.shape[1])

    def drop(self, mark):
        values = self.values.copy()             # not inplace: not change self
        if type(mark[0]) == type('string') :     # drop columns using string list; 'string'[0] is string
            mark = mark if type(mark) == type([]) else [mark]          # make it feasible for one column
            labels = np.array(self.labels)
            column = []
            for label in mark:
                column = column + [i for i, place in enumerate(self.labels) if place == label]
            values = np.delete(values, column, 1)           # delete values
            labels = np.delete(labels, column)
            labels = labels.tolist()
            sub_dataframe = DataFrame(values.squeeze(), self.index, labels) # index is not changed
            return sub_dataframe
        elif type(mark[0]) == type(0) or type(mark[0]) == type(0.0) or \
                type(mark) == type(0) or type(mark) == type(0.0):  # drop rows
            mark = mark if type(mark[0]) == type(0) or type(mark[0]) == type(0.0) else [mark]  # make it feasible for one column
            index = np.array(self.index)
            row = []
            for label in mark:
                row = row + [i for i, place in enumerate(self.index) if place == label]
            values = np.delete(values, row, 0)          # delete values
            index = np.delete(index, row)              # delete index
            index = index.tolist()
            sub_dataframe = DataFrame(values.squeeze(), index, self.labels)
            return sub_dataframe

    def loc(self, position):    # loc()operation
        if type(position) == type(0) or type(position) == type(0.0) or type(position) == type([]):
            position = np.array(position)
            position = position - 100000
            values = self.father_pointer.values[position]   # for number position
            temp = np.array(self.father_pointer.index)
        elif type(position) == np.ndarray and type(position[0]) == np.int64:
            position = position - 100000
            values = self.father_pointer.values[position]   # for number position
            temp = np.array(self.father_pointer.index)
        elif type(position) == np.ndarray and type(position[0]) == np.bool_:
            values = self.values[position]
            temp = np.array(self.index)
        index = temp[position].tolist()       # store the picked index relative to father
        sub_dataframe = DataFrame(values.squeeze(), index, self.labels, self.father_pointer)
        return sub_dataframe

    def value_counts(self, label):     # only for a column counts
        count_num = []
        value = self.__getitem__(label)
        key = np.unique(value) # find unique value
        for k in key:
            mask = (value==k)
            y_new = value[mask]
            count_num.append(y_new.shape[0])
        key = key.tolist()
        order_dict = dict(zip(count_num, key))
        count_num = sorted(count_num,reverse=True) # use dictionary to build the order of index
        for i in range(len(key)):
            key[i] = order_dict[count_num[i]]
        return DataFrame(np.array(count_num).squeeze(), key, [label])

# -----------   helpers            ---------- #

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    labels = open(data_path,'r').readline()


    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]
    labels = labels.strip().split(",")
    del labels[0]
    del labels[0]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids, labels

def log_process(x):
    x_log = np.log10(1 / (1 + x[:,sum(x<0)==0])) # to avoid log(0)
    # x_log = np.log(x[:,sum(x<0)==0])
    x = np.hstack((x, x_log))
    return x

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

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def predict_labels(weights, data):
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def build_polynomial_features(x, degree):
    temp_dict = {}
    count = 0
    for i in range(x.shape[1]):
        for j in range(i+1,x.shape[1]):
            temp = x[:,i] * x[:,j]
            temp_dict[count] = [temp]
            count += 1
    poly_length = x.shape[1] * (degree + 1) + count# + 1
    poly = np.zeros(shape = (x.shape[0], poly_length))
    for deg in range(1,degree+1):
        for i in range(x.shape[1]):
            poly[:,i + (deg-1) * x.shape[1]] = np.power(x[:,i],deg)
    for i in range(count):
        poly[:, x.shape[1] * degree + i] = temp_dict[i][0]
    for i in range(x.shape[1]):
        poly[:,i + x.shape[1] * degree + count] = np.abs(x[:,i])**0.5
    return poly            
            
# ----------    Cost calculation    ---------- #
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

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
# def calculate_logi_loss(y, tx, w):
#     """compute the cost by negative log likelihood."""
#     pred = sigmoid(tx.dot(w))
#     loss = y.T.dot(np.log(pred + 1e-12)) + (1 - y).T.dot(np.log(1 - pred + 1e-12))
#     return np.squeeze(- loss)

def compute_loss(y, tx, w, method = calculate_mse):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)   # 一维的np.array都视作'列'向量: 从其尺寸(2000,)也能看出，2000行，列向量
    return method(e)
    # return calculate_mae(e)

def compute_accuracy(y_pred,y_true):
    return (1 - sum(abs(y_pred-y_true)/2)/len(y_pred))
    
# ----------     cross validation   ---------- #

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, regression_method, **kwargs):
    
    test_idx = k_indices[k]
    train_idx = list(set(np.arange(0,len(y)))-set(k_indices[k]))
    [x_train,y_train,x_test,y_test] = [x[train_idx], y[train_idx], x[test_idx], y[test_idx]]
    
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    loss,weight = regression_method(y = y_train, tx = x_train, **kwargs)
    
    # loss_tr = np.sqrt(2 * compute_mse(y_train,x_train,weight))
    # loss_te = np.sqrt(2 * compute_mse(y_test,x_test,weight))    

    y_train_pred = predict_labels(weight, x_train)
    y_test_pred = predict_labels(weight, x_test)

    accuracy_train = compute_accuracy(y_train_pred, y_train)
    accuracy_test = compute_accuracy(y_test_pred, y_test)

    return weight,accuracy_train, accuracy_test   

def cv_loop(y, x, k_fold, seed, regression_method, **kwargs):
    k_indices = build_k_indices(y, k_fold, seed)
    weight = np.zeros(x.shape[1]+1)
    list_accuracy_train = []
    list_accuracy_test = []
    
    for k in range(k_fold):
        w,acc_tr,acc_te = cross_validation(y, x, k_indices, k, regression_method, **kwargs)
        weight = weight + w
        list_accuracy_train.append(acc_tr)
        list_accuracy_test.append(acc_te)
        # print("{} fold cv: Training accuracy: {} - Test accuracy : {}".format(k, acc_tr, acc_te))
    
    return weight/k_fold,np.mean(list_accuracy_train),np.mean(list_accuracy_test)
    
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

# def ridge_regression(y, x):
#     """ridge regression demo."""
#     # define parameter
#     lambdas = np.logspace(-15, 5, 50)
#     rmse_tr = []
#     for ind, lambda_ in enumerate(lambdas):
#         # ridge regression
#         weight = ridge_regression_solve(y, x, lambda_)
#         rmse_tr.append(np.sqrt(2 * compute_loss(y, x, weight)))
#         print("lambda={l:.3f}, Training RMSE={tr:.3f}".format(l=lambda_, tr=rmse_tr[ind]))


# ----------    Logistic regression    ---------- #
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

def logistic_regression_GD(y, tx, max_iters, gamma):
    # init parameters
    losses = []
    w = np.zeros((tx.shape[1], 1))
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

data_path = 'data/train.csv'
data_label, data_origin, _, _ = load_csv_data(data_path, sub_sample=False)
data_standardized, data_origin_mean ,data_origin_std = standardize(data_origin)
# max_iters = 10000
# threshold = 1e-8
# gamma = 0.01
# #w_initial = np.array([0]*data_standardized.shape[1], dtype='float64')
# logistic_regression_GD(data_label, data_standardized,max_iters, gamma)

w_initial = np.array([0]*data_standardized.shape[1])
max_iters = 100
gamma = 0.0007
gradient_losses, gradient_ws = gradient_descent(data_label, data_standardized, w_initial, max_iters, gamma)
