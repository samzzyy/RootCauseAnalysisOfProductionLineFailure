import pandas as pd
import corrAnaModule as cam
import numpy as np
pd.options.mode.chained_assignment = None

def count_ratio_every_col_obj(df_raw:pd.DataFrame):
    new_col=[]
    total=df_raw.shape[0]
    for _ in df_raw.columns:
        a=pd.value_counts(df_raw[_])
        df_=a/total
        # print(_,max(df_))
        if (max(df_)<0.95):
            new_col.append(_)
    return new_col

def data_generation():
    df_equip_history=pd.read_csv("F:\\YIELD\\YoudaOptronics\\Archive(1)\\equip_history.csv",engine="python",sep=',',encoding='GBK')
    df_equip_history['SHEET_ID']=df_equip_history['锘縎HEET_ID'] # modify unidentifiable columns
    df_equip_history.drop(columns=['锘縎HEET_ID'],inplace=True)
    df_equip_history.fillna('-1',inplace=True)

    col_list=[]
    for col in list(df_equip_history.columns):
        if 'R' not in col:
            col_list.append(col)

    df_measure_labels=pd.read_csv("F:\\YIELD\\YoudaOptronics\\Archive(1)\\measure_labels.csv",engine="python",sep=',',encoding='GBK')
    df_measure_labels.dropna(inplace=True)

    df_temp=pd.merge(df_equip_history[col_list],df_measure_labels[['SHEET_ID','Y']],how='inner',on='SHEET_ID')
    df_temp['label']=0
    df_temp['label'][(df_temp['Y']>=1)|(df_temp['Y']<=-1)]=1
    col_list.remove('SHEET_ID')


    df_label1=df_temp[df_temp['label']==1].copy()#sheetId有大量重复，这里选择只要有出现过不良的就作为响应标签
    df_label1.drop_duplicates(inplace=True)
    df_label0=df_temp[df_temp['label']==0].copy()
    df_label0.drop_duplicates(inplace=True)
    df_temp=pd.concat([df_label1,df_label0],axis=0)
    df_temp.drop_duplicates('SHEET_ID','first',inplace=True)

    columns_list=count_ratio_every_col_obj(df_temp)
    columns_list.remove('SHEET_ID')
    columns_list.remove('Y')
    return df_temp[columns_list]

def correlation_index_rank(df_cluster,corr_funciton_name):

    corr_funciton=getattr(cam, corr_funciton_name)
    columns_list = list(df_cluster.columns)
    columns_list.remove('label')
    corr_index_list=[]
    for _ in columns_list:
        corr_temp=corr_funciton(df_cluster,_)
        corr_index_list.append(corr_temp)

    df_col_rank=pd.DataFrame({'col':columns_list,corr_funciton_name[:-6]:corr_index_list})
    df_col_rank.sort_values(by=corr_funciton_name[:-6],ascending=False,inplace=True)
    df_col_rank.index=range(df_col_rank.shape[0])
    # print(df_col_rank)
    return df_col_rank

def final_rank_confidence(df_cluster,corr_func_list):
    df_rank = pd.DataFrame({'col': list(df_cluster.columns)})
    df_rank['time_index'] = df_rank.index + 1
    for func in corr_func_list:
        df_rank_temp = correlation_index_rank(df_cluster, corr_funciton_name=func)
        df_rank_temp[func] = df_rank_temp.index + 1
        df_rank = pd.merge(df_rank, df_rank_temp[['col',func]], on='col')
    rank_col_list = list(df_rank.columns)
    rank_col_list.remove('col')
    r = len(rank_col_list)

    df_rank['final_rank']=1#计算 R1*R2*..Rn开n次根
    for _ in rank_col_list:
        df_rank['final_rank'] = df_rank['final_rank']*df_rank[_]
    df_rank['final_rank']=pow(df_rank['final_rank'], 1 / r)
    df_rank.sort_values('final_rank', ascending=True, inplace=True)
    df_rank.index=range(df_rank.shape[0])
    print(df_rank)
    return df_rank

df_cluster = data_generation()
final_rank_confidence(df_cluster,['iv_index','chi_square_index'])

# gini_index=getattr(cam, 'gini_index')
#
# for _ in
# df_rank=gini_index(df_cluster,)