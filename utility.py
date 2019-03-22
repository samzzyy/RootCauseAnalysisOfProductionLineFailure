import numpy as np
import pandas as pd
from sklearn import preprocessing

#连续&名义
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm #方差分析
#
# # df_data=pd.DataFrame({"c1":[1,1,3,2,4,1,2,2, 2,2,3,4,3,1,2,1 ,3],
# #                       "c2":[1,2,3,4,3,1,3,1, 3,2,1,3,4,1,2,3 ,3],
# #                       "c5":[1,3,4,2,3,3,3,2, 4,2,2,3,3,2,2,1 ,3],
# #                       "c4":[1,2,2,4,1,3,3,4, 3,2,1,3,2,4,2,3 ,3],
# #                       "c3":[1,2,6,4,2,6,6,2, 6,1,2,3,6,6,1,2 ,3],
# #                       "c6":[1,2,3,4,1,4,4,1, 2,3,1,4,4,1,4,1 ,3],
# #                       "label":[0,0,1,0,0,1,1,0, 1,0,0,0,1,1,0,0,1]})
#
# df_data=pd.DataFrame({"c1":[1,1,3,2,4,1,2,2, 2,2,3,4,3,1,2,1 ,3,5,6,7,8,7,7,6, 8,7,6,6,7,8,5,5],
#                       "c2":[1,2,3,4,3,1,3,1, 3,2,1,3,4,1,2,3 ,3,8,7,5,6,6,7,8, 7,8,7,7,6,5,6,8],
#                       "c5":[1,3,4,2,3,3,3,2, 4,2,2,3,3,2,2,1 ,3,7,8,7,7,6,5,6, 8,5,6,7,8,7,7,6,],
#                       "c4":[1,2,2,4,1,3,3,4, 3,2,1,3,2,4,2,3 ,3,8,7,6,6,7,8,5, 5,8,7,7,6,5,6,8],
#                       "c3":[1,2,9,4,2,9,9,2, 9,1,2,3,9,9,1,2 ,3,7,8,9,7,9,5,6, 8,6,9,5,6,7,9,7],
#                       "c6":[1,2,3,4,1,4,4,1, 2,3,1,4,4,1,4,1 ,3,7,5,6,6,7,8,7, 9,8,7,5,6,6,7,8],
#                       "label":[0,0,1,0,0,1,1,0, 1,0,0,0,1,1,0,0, 0,0,0,0,1,0,1,0, 0,0,0,1,0,0,1,0 ]})
#
# model=ols("label~c1+c2+c5+c4+c3+c6",df_data).fit()
# anovad = anova_lm(model)
# print(anovad)


# from scipy.stats import f # F检验
#
# columns_list=list(df_data.columns)
# columns_list.remove("label")
# b=df_data["label"]
# for _ in columns_list:
#     a=df_data[_]
#     F = np.var(a) / np.var(b)
#     df1 = len(a) - 1
#     df2 = len(b) - 1
#     p_value = 1 - 2 * abs(0.5 - f.cdf(F, df1, df2))
#     print(_, p_value)

#名义&名义

#基尼方差

#皮尔森卡方统计量
from scipy.stats import gamma
df_data=pd.DataFrame({"c1":[1,0,1,0,0,1,0,1, 1,0,0,0,0,1,1,1],
                      "c2":[1,0,0,0,1,0,0,0, 0,0,1,1,1,1,0,0],
                      "c5":[0,1,1,1,0,1,1,1, 1,1,1,0,1,0,1,1],
                      "c4":[1,0,1,0,1,0,0,1, 0,1,0,1,1,0,1,1],
                      "c3":[1,1,1,1,0,1,0,0, 1,0,1,0,1,0,0,1],
                      "c6":[1,0,0,0,0,1,1,1, 1,0,1,1,1,1,0,1],
                      # "c7":[1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1],
                      'c8':[1,0,0,0,0,1,0,0, 1,0,1,0,1,0,0,1],
                      "label":[1,0,0,0,0,1,0,0, 1,0,1,0,1,0,0,1]})
columns_list=list(df_data.columns)
columns_list.remove('label')
feature_data=df_data[columns_list]
df=1 #卡方分布的自由度

N=df_data.shape[0]#
for _ in columns_list:
    miu_list=[]
    n_ij_list=[]
    for i in range(2):
        for j in range(2):
            n_i_=df_data[_][df_data[_]==i].shape[0]
            n__j=df_data[_][df_data['label']==j].shape[0]
            miu_ij = (n_i_*n__j)/ N
            miu_list.append(miu_ij)

            n_ij = df_data[(df_data[_] == i) & (df_data['label'] == j)].shape[0]
            n_ij_list.append(n_ij)

    X_2=sum((np.array(n_ij_list)-np.array(miu_list))**2/np.array(miu_list))
    p_value = 1 - gamma.cdf(X_2, df)
    print(_,p_value)



