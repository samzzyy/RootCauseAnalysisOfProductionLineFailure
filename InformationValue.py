import math
import numpy as np
import pandas as pd

def discrete_sampleNumMean(x,discrete_num):
    '''
    Discrete the input 1-D numpy array using 5 equal percentiles
    :param x: 1-D numpy array
    :return: discreted 1-D numpy array
    '''
    res = np.zeros(x.shape[0], dtype=int)
    x_sort_indice = np.argsort(x)
    rate_discrete = 1 / discrete_num
    for i in range(discrete_num):
        point1 = int(x.shape[0] * i * rate_discrete)
        point2 = int(x.shape[0] * (i + 1) * rate_discrete)
        res[x_sort_indice[range(point1, point2)]] = (i + 1)
    return res

def feature_discretion(X,discrete_num):
    '''
    Discrete the continuous features of input data X, and keep other features unchanged.
    :param X : numpy array
    :return: the numpy array in which all continuous features are discreted
    '''
    temp = []
    for i in range(0, X.shape[-1]):
        x = X[:, i]
        x1 = discrete_sampleNumMean(x,discrete_num)
        temp.append(x1)
    return np.array(temp).T

def count_binary(a, event=1):
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count

def woe_single_x(x, y, event=1):
        '''
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        _WOE_MIN = -20
        _WOE_MAX = 20
        event_total, non_event_total = count_binary(y, event=event)
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = count_binary(y1, event=event)
            print('1:',event_count,'0:',non_event_count)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = _WOE_MIN
            elif rate_non_event == 0:
                woe1 = _WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

def woe( X, y, event=1,discrete_num=4):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        X1 = feature_discretion(X,discrete_num)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):#X1.shape[-1]= X1.shape[1]
            print('第%d号特征'%i)
            x = X1[:, i]
            woe_dict, iv1 = woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

df_data=pd.DataFrame({"c1":[1,1,3,2,4,1,2,2, 2,2,3,4,3,1,2,1],
                      "c2":[1,2,3,4,3,1,3,1, 3,2,1,3,4,1,2,3],
                      "c5":[1,3,4,2,3,3,3,2, 4,2,2,3,3,2,2,1],
                      "c4":[1,2,2,4,1,3,3,4, 3,2,1,3,2,4,2,3],
                      "c3":[1,2,6,4,2,6,6,2, 6,1,2,3,6,6,1,2],
                      "label":[0,0,1,0,0,1,1,0, 1,0,0,0,1,1,0,0]})
df_data=pd.DataFrame({"c1":[1,0,1,0,0,1,0,1, 1,0,0,0,0,1,1,1],
                      "c2":[1,0,0,0,1,0,0,0, 0,0,1,1,1,1,0,0],
                      "c5":[0,1,1,1,0,1,1,1, 1,1,1,0,1,0,1,1],
                      "c4":[1,0,1,0,1,0,0,1, 0,1,0,1,1,0,1,1],
                      "c3":[1,1,1,1,0,1,0,0, 1,0,1,0,1,0,0,1],
                      "c6":[1,0,0,0,0,1,1,1, 1,0,1,1,1,1,0,1],
                      "label":[1,0,0,0,0,1,0,0, 1,0,1,0,1,0,0,1]})
data=df_data[df_data.columns[:-1]].values
label=np.array(df_data["label"])
woe,iv=woe(data,label)
print(woe,iv)
for iv_temp in iv:
    print(iv_temp)