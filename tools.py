import numpy as np
import csv

# ----------    Simple DataFrame class replace the function of pandas.DataFrame---------- #
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

# -----------   data processing            ---------- #

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

def compute_accuracy(y_pred,y_true):
    return (1 - sum(abs(y_pred-y_true)/2)/len(y_pred))
    
# ----------     cross validation   ---------- #
def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

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
    
    loss,weight = regression_method(y = y_train, tx = x_train, **kwargs)
    
    y_train_pred = predict_labels(weight, x_train)
    y_test_pred = predict_labels(weight, x_test)

    accuracy_train = compute_accuracy(y_train_pred, y_train)
    accuracy_test = compute_accuracy(y_test_pred, y_test)

    return weight,accuracy_train, accuracy_test   

def cv_loop(y, x, k_fold, seed, regression_method, **kwargs):
    k_indices = build_k_indices(y, k_fold, seed)
    weight = np.zeros(x.shape[1])
    list_accuracy_train = []
    list_accuracy_test = []
    
    for k in range(k_fold):
        w,acc_tr,acc_te = cross_validation(y, x, k_indices, k, regression_method, **kwargs)
        weight = weight + w
        list_accuracy_train.append(acc_tr)
        list_accuracy_test.append(acc_te)
        # print("{} fold cv: Training accuracy: {} - Test accuracy : {}".format(k, acc_tr, acc_te))
    
    return weight/k_fold,np.mean(list_accuracy_train),np.mean(list_accuracy_test)

# ----------     Grid Search   ---------- #

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
