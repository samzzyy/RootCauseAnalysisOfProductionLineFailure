import numpy as np
import pandas as pd
from collections import defaultdict

def chi_square_index(df_cluster : pd.DataFrame, attr_name):
    N=df_cluster.shape[0]
    yi_dict = defaultdict(int)  # 每个特征，每种值中label =1的数量
    ni_dict = defaultdict(int)  # 每个特征，每种值 label =0的数量

    for _ in df_cluster[attr_name].unique():
        yi_dict[_] = df_cluster[(df_cluster[attr_name] == _) & (df_cluster['label'] == 1)].shape[0]
        ni_dict[_] = df_cluster[(df_cluster[attr_name] == _) & (df_cluster['label'] == 0)].shape[0]

    yi_arr = np.array(list(yi_dict.values()))
    ni_arr = np.array(list(ni_dict.values()))
    yi_arr[yi_arr == 0] = 1
    ni_arr[ni_arr == 0] = 1  # Laplace平滑，令概率不为0

    chi_table=np.concatenate((yi_arr,ni_arr)).reshape((2,-1))
    x22=0
    for i in range(chi_table.shape[0]):
        for j in range(chi_table.shape[1]):
            u_ij=sum(chi_table[i])*sum(chi_table[:,j])/N
            x2_temp=(chi_table[i,j]-u_ij)**2/u_ij
            x22=x22+x2_temp
    return x22

def iv_index(df_cluster : pd.DataFrame, attr_name):
    """
    :param cluster:
    :param attr_name:
    :return:
    """
    yi_dict = defaultdict(int)  # 每个特征，每种值中label =1的数量
    ni_dict = defaultdict(int)  # 每个特征，每种值 label =0的数量
    label_t_dict = defaultdict(int)  # label=1或0的数量

    label_t_dict[0]=df_cluster[df_cluster['label']==0].shape[0]
    label_t_dict[1]=df_cluster[df_cluster['label']==1].shape[0]

    for _ in df_cluster[attr_name].unique():
        yi_dict[_]=df_cluster[(df_cluster[attr_name]==_) & (df_cluster['label']==1) ].shape[0]
        ni_dict[_]=df_cluster[(df_cluster[attr_name]==_) & (df_cluster['label']==0) ].shape[0]

    yi_arr = np.array(list(yi_dict.values()))
    ni_arr = np.array(list(ni_dict.values()))

    yi_arr[yi_arr == 0] = 1
    ni_arr[ni_arr == 0] = 1  # Laplace平滑，令概率不为0
    pyi = yi_arr / label_t_dict[1]
    pni = ni_arr / label_t_dict[0]

    pyi_div_pni = pyi / pni
    iv_i_arr = ((pyi) - (pni)) * np.log(pyi_div_pni)
    return sum(iv_i_arr)


def gini_index(df_cluster : pd.DataFrame, attr_name):
    def gini(D):
        """求基尼指数 Gini(D)
        :param D: shape = [ni_samples]
        :return: Gini(D)
        """
        # 目前版本的 numpy.unique 不支持 axis 参数
        _, cls_counts = np.unique(D, return_counts=True)
        probability = cls_counts / cls_counts.sum()
        return 1 - (probability ** 2).sum()

    def congini(D_, val):
        """求基尼指数 Gini(D, A)
        :param D_: 被计算的列. shape=[ni_samples, 2]
        :param val: 被计算的列对应的切分变量
        :return: Gini(D, A)
        """
        left, right = D_[D_[:, 0] == val], D_[D_[:, 0] != val]
        # print(val,left.shape[0],right.shape[0])
        return gini(left[:, -1]) * left.shape[0] / D_.shape[0] + \
               gini(right[:, -1]) * right.shape[0] / D_.shape[0]
    D_=df_cluster[[attr_name,'label']].values
    unique_val_list=pd.unique(df_cluster[attr_name])
    gini_by_A_list=[]
    for _ in unique_val_list:
        gini_by_A_list.append(congini(D_,_))
    # print(attr_name,gini_by_A_list,unique_val_list)
    return max(gini_by_A_list)
