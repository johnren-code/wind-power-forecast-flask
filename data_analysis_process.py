import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest

import random
seed = 42
np.random.seed(seed)
random.seed(seed)

import warnings
warnings.filterwarnings('ignore')

def cal_miss_info(file_path):
    """缺失值统计"""
    data=pd.read_csv(file_path)
    miss_count = data.isnull().sum().sort_values(ascending=False)
    miss_pert = miss_count / len(data)
    miss_info = pd.concat([miss_count, miss_pert], axis=1, keys=["缺失计数", "缺失百分比"])
    return miss_info.to_json()

def descriptive_analysis(file_path):
    """描述性统计，返回的是一个DataFrame"""
    df = pd.read_csv(file_path)
    res = df.describe()
    res.index = ['样本量','平均值','标准差','最小值','25','中位数','75','最大值']
    res.drop(index=['25', '75'],inplace=True)
    res.reindex(['样本量','最大值','最小值','平均值','标准差','中位数'])
    res = pd.concat([res, pd.DataFrame(pd.Series(df.var(), name='方差')).T])
    res = pd.concat([res, pd.DataFrame(pd.Series(df.kurt(), name='峰度')).T])
    res = pd.concat([res, pd.DataFrame(pd.Series(df.skew(), name='偏度')).T])
    res = pd.concat([res, pd.DataFrame(pd.Series(res.loc['标准差']/res.loc['平均值'], name='变异系数(CV)')).T])
    return res.to_json()

def correlation_analysis(file_path):
    """相关性分析"""
    df = pd.read_csv(file_path)
    correlation_matrix = df.corr()
    # 转换为指定格式的列表
    heatmap_data = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            heatmap_data.append([i, j, float(f'{value:.2f}')])
    return correlation_matrix.to_json()


#=========================异常检测之前进行填充的函数============================
def KNN_Impute(df):
    # 异常检测之前的KNN填充
    numeric_columns = ['WINDSPEED','PREPOWER','WINDDIRECTION','TEMPERATURE','HUMIDITY','PRESSURE','ROUND(A.WS,1)','ROUND(A.POWER,0)','YD15']
    imputer = KNNImputer(n_neighbors=500, weights='distance')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

def KNN_RF_Impute(df):
    # 异常检测之前的KNN+RF混合填充
    features = ['WINDSPEED','PREPOWER','WINDDIRECTION','TEMPERATURE','HUMIDITY','PRESSURE','ROUND(A.WS,1)']
    imputer = KNNImputer(n_neighbors=500, weights='distance')
    df[features] = imputer.fit_transform(df[features])
    targets = ['ROUND(A.POWER,0)', 'YD15']
    RF = RandomForestRegressor()
    # 遍历目标列
    for column in targets:
        # 划分已知值和待填充值
        known = df[df[column].notnull()]  # 已知值的子集
        unknown = df[df[column].isnull()]  # 待填充值的子集
        # 拆分已知值的特征和目标
        X_known = known[features]
        y_known = known[column]
        # 拟合模型
        RF.fit(X_known, y_known)
        # 预测待填充值
        X_unknown = unknown[features]
        y_pred = RF.predict(X_unknown)
        # 填充空缺值
        nan_index = df[column].isnull()
        df.loc[nan_index, column] = y_pred
    return df


#=========================异常检测可选择的函数============================
def IF_Detection(df):
    # 孤立森林检测
    df = df.set_index('DATATIME')
    IF = IsolationForest(contamination=0.25)  # 设置异常值比例
    numeric_columns = ['WINDSPEED','PREPOWER','WINDDIRECTION','TEMPERATURE','HUMIDITY','PRESSURE','ROUND(A.WS,1)','ROUND(A.POWER,0)','YD15']
    IF.fit(df[numeric_columns])
    outliers = IF.predict(df[numeric_columns])
    nan_index = outliers == -1
    sns.scatterplot(x=df.loc[nan_index,'ROUND(A.WS,1)'],y=df.loc[nan_index,'YD15'])
    df.loc[nan_index, ['ROUND(A.POWER,0)','YD15']] = np.nan
    return df

def LOF_Detection(df):
    # LOF检测（基于密度度量）
    from sklearn.neighbors import LocalOutlierFactor
    numeric_columns = ['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']
    df = df.set_index('DATATIME')
    LOF = LocalOutlierFactor(n_neighbors=20, contamination=0.25)  # 设置邻居数和异常值比例
    outlier_scores = LOF.fit_predict(df[numeric_columns])
    nan_index = outlier_scores == -1
    df.loc[nan_index, ['ROUND(A.POWER,0)', 'YD15']] = np.nan
    return df


# def DeepLog_Detection(df):
#     # DeepLog检测（深度学习：基于自动编码器）
#     from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
#     numeric_columns = ['WINDSPEED','PREPOWER','WINDDIRECTION','TEMPERATURE','HUMIDITY','PRESSURE','ROUND(A.WS,1)','ROUND(A.POWER,0)','YD15']
#     df = df.set_index('DATATIME')
#     transformer = DeepLogSKI(features=len(df.columns), contamination=0.1,epochs=10,stacked_layers=3)
#     transformer.fit(np.array(df))
#     prediction_labels = transformer.predict(np.array(df))
#     prediction_score = transformer.predict_score(np.array(df))
#     nan_index = (prediction_labels == 1).squeeze()
#     df.loc[nan_index, ['ROUND(A.POWER,0)','YD15']] = np.nan
#     return df
#
#
# def Telemanom_Detection(df):
#     # Telemanom检测（深度学习：基于循环神经网络）
#     from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
#     df = df.set_index('DATATIME')
#     l_s = 96
#     n_predictions = 4
#     out_ratio = 0.15
#     transformer = TelemanomSKI(l_s= l_s, n_predictions= n_predictions,contamination=out_ratio)
#     transformer.fit(np.array(df))
#     prediction_labels = transformer.predict(np.array(df))
#     prediction_score = transformer.predict_score(np.array(df))
#     nan_index = np.hstack([np.array([False]*(l_s+n_predictions)),(prediction_labels == 1).squeeze()])
#     df.loc[nan_index, ['ROUND(A.POWER,0)','YD15']] = np.nan
#     return df




#=========================异常检测之后进行填充的函数============================
def KNN_Impute_after(df):
    # 异常检测之后的KNN填充
    numeric_columns = ['WINDSPEED','PREPOWER','WINDDIRECTION','TEMPERATURE','HUMIDITY','PRESSURE','ROUND(A.WS,1)','ROUND(A.POWER,0)','YD15']
    imputer = KNNImputer(n_neighbors=100, weights='distance')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

def RF_Impute_after(df):
    # 异常检测之后的随机森林填充
    df['ROUND(A.WS,1)']=df['ROUND(A.WS,1)'].interpolate(method='linear', limit_direction='both')
    features = ['WINDSPEED','PREPOWER','WINDDIRECTION','TEMPERATURE','HUMIDITY','PRESSURE','ROUND(A.WS,1)']
    targets = ['ROUND(A.POWER,0)', 'YD15']

    RF = RandomForestRegressor()
    # 遍历目标列
    for column in targets:
        # 划分已知值和待填充值
        known = df[df[column].notnull()]  # 已知值的子集
        unknown = df[df[column].isnull()]  # 待填充值的子集
        # 拆分已知值的特征和目标
        X_known = known[features]
        y_known = known[column]
        # 拟合模型
        RF.fit(X_known, y_known)
        # 预测待填充值
        X_unknown = unknown[features]
        y_pred = RF.predict(X_unknown)
        # 填充空缺值
        nan_index = df[column].isnull()
        df.loc[nan_index, column] = y_pred
        sns.scatterplot(x=df.loc[nan_index,'ROUND(A.WS,1)'],y=y_pred)
    return df

def data_preprocess(file_path,before_impute_method,detection,after_impute_method):
    """数据预处理
    1. 根据时间排序，并去除重复样本
    2. 处理离群样本点
    3. 处理一天内YD15持续不变的样本点
    4. 填充缺失值
    5. 纠正异常值
    """

    # 初步处理
    df = pd.read_csv(file_path)
    df = df.sort_values(by='DATATIME', ascending=True)
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    df.reset_index(drop=True,inplace=True)
    columns_to_keep = ['TurbID','DATATIME','WINDSPEED','PREPOWER','WINDDIRECTION','TEMPERATURE','HUMIDITY','PRESSURE','ROUND(A.WS,1)','ROUND(A.POWER,0)','YD15']
    numeric_columns = ['WINDSPEED','PREPOWER','WINDDIRECTION','TEMPERATURE','HUMIDITY','PRESSURE','ROUND(A.WS,1)','ROUND(A.POWER,0)','YD15']
    df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])



    # 找出离群点并设为空值
    out_cols = ['ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']
    for col in out_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        outliers_index = df[outliers].index
        df.loc[outliers_index, col] = np.nan



    # 找出一天内持续不变的YD15设为空值
    df['DATE'] = df['DATATIME'].dt.date
    grouped = df.groupby(['DATE'])
    unique_counts = grouped['YD15'].nunique()
    stale_rows = unique_counts[unique_counts == 1].index.tolist()
    stale_indices = []
    for date in stale_rows:
        indices = df.index[(df['DATE'] == date)]
        stale_indices.extend(indices.tolist())
    df.loc[stale_indices, 'YD15'] = np.nan
    df.drop(columns=['DATE'], inplace=True)

    # 将时间列转换为 datetime 类型
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])


    # 特殊异常值处理
    df.loc[df['ROUND(A.WS,1)']<0, 'ROUND(A.WS,1)'] = np.nan
    df.loc[df['ROUND(A.WS,1)']>1e6, 'ROUND(A.WS,1)'] = np.nan
    columns = ['ROUND(A.POWER,0)','YD15']
    for col in columns:
        df.loc[(df[col]>1e6) | (df[col]<-1e6), col]=np.nan
        df.loc[(df['ROUND(A.WS,1)']==0) & (df[col]>0), col] = 0
        df.loc[(df['ROUND(A.WS,1)']>20) & (df[col]!=0), col] = 0
        df.loc[(df['ROUND(A.WS,1)']>5) & (df[col]==0), col] = np.nan



    # 检测异常值之前填充
    if before_impute_method=='KNN':
        df=KNN_Impute(df)
    elif before_impute_method=='KNN_RF':
        df=KNN_RF_Impute(df)

    # 检测异常值
    if detection=='IF':
        df=IF_Detection(df)
    elif detection=='LOF':
        df=LOF_Detection(df)
    # elif detection=='Telemanom':
    #     df=Telemanom_Detection(df)
    # elif detection=='DEEPLOG':
    #     df=DeepLog_Detection(df)



    # 检测异常值之后填充
    if after_impute_method=='KNN':
        df=KNN_Impute_after(df)
    elif after_impute_method=="RF":
        df=RF_Impute_after(df)

    save_path="../processed_data/n"+file_path.split('/')[len(file_path.split('/'))-1]
    df.to_csv(save_path)
    return save_path
