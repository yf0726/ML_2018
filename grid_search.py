import numpy as np

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1

def get_best_parameters(para1,para2, acc):
    """Get the best w from the result of grid search."""
    max_row, max_col = np.unravel_index(np.argmax(acc), acc.shape)
    return acc[max_row, max_col], para1[max_row], para2[max_col]

def grid_search(y, tx, para1, para2):
    """Algorithm for grid search."""
    acc_tr = np.zeros((len(para1), len(para2)))
    acc_te = np.zeros((len(para1), len(para2)))
    tx = log_process(tx)
    k_indices = build_k_indices(y, 10, seed = 1)
    for i in range(0,len(para1)):
        for j in range(0,len(para2)):
            tx_poly = build_poly(tx, para1[i])
            tx_poly,_,_ = standardize(tx_poly)
            _,acc_tr[i][j],acc_te[i][j] = cross_validation(y, tx_poly, k_indices, 0, para2[j])
    return acc_tr,acc_te
# ***************************************************

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_):
    
    test_idx = k_indices[k]
    train_idx = list(set(np.arange(0,len(y)))-set(k_indices[k]))
    [x_train,y_train,x_test,y_test] = [x[train_idx], y[train_idx], x[test_idx], y[test_idx]]
    
    loss,weight = ridge_regression(y = y_train, tx = x_train, lambda_=lambda_)
    
    y_train_pred = predict_labels(weight, x_train)
    y_test_pred = predict_labels(weight, x_test)

    accuracy_train = compute_accuracy(y_train_pred, y_train)
    accuracy_test = compute_accuracy(y_test_pred, y_test)

    return weight,accuracy_train, accuracy_test   

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    return loss,w

def log_process(x):
    """"""
    x_log = np.log10(1 + x[:,sum(x<0)==0]) # to avoid log(0)
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
    
    x_standardize = np.hstack((np.ones((x_standardize.shape[0], 1)), x_standardize))

    return x_standardize, mean_x, std_x

def build_poly(x, degrees):
    tmp_dict = {}
    count = 0
    for i in range(x.shape[1]):
        for j in range(i+1,x.shape[1]):
            tmp = x[:,i] * x[:,j]
            tmp_dict[count] = [tmp]
            count += 1
    poly_len = x.shape[1] * (degrees + 1) + count
    poly = np.zeros((x.shape[0], poly_len))
    for degree in range(1,degrees+1):
        for i in range(x.shape[1]):
            poly[:,i + (degree-1) * x.shape[1]] = np.power(x[:,i],degree)
    for i in range(count):
        poly[:, x.shape[1] * degrees + i] = tmp_dict[i][0]
    for i in range(x.shape[1]):
        poly[:,i + x.shape[1] * degrees + count] = np.sqrt(np.abs(x[:,i]))
    return poly            


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

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def compute_accuracy(y_pred,y_true):
    return (1 - sum(abs(y_pred-y_true)/2)/len(y_pred))

