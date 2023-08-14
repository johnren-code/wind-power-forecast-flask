import json

import numpy as np
import pandas as pd
import seaborn as sns
from flask import jsonify
from pyecharts.render import make_snapshot, snapshot

from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts
from pyecharts.globals import ThemeType, SymbolType
from pyecharts.charts import Bar, HeatMap, Grid, Boxplot, Scatter3D, Scatter
import pyecharts.options as opts
from pyecharts.faker import Faker


import random

seed = 42
np.random.seed(seed)
random.seed(seed)

import warnings

warnings.filterwarnings("ignore")

# def descriptive_table(file_path):
#     table = Table()
#     df = descriptive_analysis(file_path).round(2)
#     df.loc["方差"] = df.loc["方差"].map(lambda x: "{:.2g}".format(x))
#     df["指标"] = df.index
#     df = df.reindex(
#         columns=[
#             "指标",
#             "WINDSPEED",
#             "PREPOWER",
#             "WINDDIRECTION",
#             "TEMPERATURE",
#             "HUMIDITY",
#             "PRESSURE",
#             "ROUND(A.WS,1)",
#             "ROUND(A.POWER,0)",
#             "YD15",
#         ]
#     )
#     headers = df.columns.to_list()
#     rows = df.values.tolist()
#     table.add(headers, rows, attributes={"align": "c", "float_format": ".2f"})
#     table.set_global_opts(title_opts=ComponentTitleOpts(title="基本统计量"))
#     make_snapshot(snapshot,table.render(),'./static/descriptive_table.png')
#     return './static/descriptive_table.png'

def cal_miss_info(file_path):
    """缺失值统计"""
    data = pd.read_csv(file_path)
    print('读取到了')
    miss_count = data.isnull().sum().sort_values(ascending=False)
    miss_pert = miss_count / len(data)
    miss_info = pd.concat([miss_count, miss_pert], axis=1, keys=["缺失计数", "缺失百分比"])
    return miss_info


def miss_info_bar(file_path,*,canvas_weight='1750px',canvas_height='500px'):
    miss_info = cal_miss_info(file_path).round(3)
    bar = (
        Bar(init_opts=opts.InitOpts(width=canvas_weight,height=canvas_height, theme=ThemeType.DARK))
        .add_xaxis([i for i in miss_info.index])
        .add_yaxis(
            "缺失百分比",
            miss_info["缺失百分比"].tolist(),
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(),
        )
        .reversal_axis()
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_top="bottom"),
            title_opts=opts.TitleOpts(title="缺失值统计", pos_left="center", pos_top=20),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                    restore=opts.ToolBoxFeatureRestoreOpts(is_show=False),
                ),
            ),
            xaxis_opts=opts.AxisOpts(is_show=True),
            yaxis_opts=opts.AxisOpts(
                axistick_opts=opts.AxisTickOpts(is_show=True),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(opacity=0)
                ),
                axislabel_opts=opts.LabelOpts(font_size=12, margin=8),
            ),
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=True, color="white", position="right"),
            itemstyle_opts={
                "color": {
                    "type": "linear",
                    "x": 0,
                    "y": 0,
                    "x2": 1750,
                    "y2": 0,
                    "colorStops": [
                        {"offset": 0, "color": "#555555"},
                        {"offset": 0.1, "color": "#66ccff"},
                        {"offset": 0.3, "color": "#ffff00"},
                        {"offset": 0.5, "color": "#fd9010"},
                        {"offset": 0.7, "color": "#fd1105"},
                    ],
                    "global": True,
                }
            },
        )
    )
    return bar

def descriptive_analysis(file_path):
    """描述性统计，返回的是一个DataFrame"""
    df = pd.read_csv(file_path)
    res = df.describe()
    res.index = ["样本量", "平均值", "标准差", "最小值", "25", "中位数", "75", "最大值"]
    res.drop(index=["25", "75"], inplace=True)
    res.reindex(["样本量", "最大值", "最小值", "平均值", "标准差", "中位数"])
    res = pd.concat([res, pd.DataFrame(pd.Series(df.var(), name="方差")).T])
    res = pd.concat([res, pd.DataFrame(pd.Series(df.kurt(), name="峰度")).T])
    res = pd.concat([res, pd.DataFrame(pd.Series(df.skew(), name="偏度")).T])
    res = pd.concat(
        [
            res,
            pd.DataFrame(pd.Series(res.loc["标准差"] / res.loc["平均值"], name="变异系数(CV)")).T,
        ]
    )
    for _,i in enumerate(["样本量", "最大值", "最小值", "平均值", "标准差", "中位数","方差","峰度","偏度","变异系数(CV)"]):#1:样本量，以此类推
        res.loc[_]=res.loc[i]
        res.drop(i,inplace=True)
    res['ws']=res['ROUND(A.WS,1)']
    res['power']=res['ROUND(A.POWER,0)']
    res.drop(['ROUND(A.POWER,0)','ROUND(A.WS,1)'],axis=1,inplace=True)
    return res.round(3).to_json(double_precision=6)


def plot_cor_matrix(file_path,*,canvas_width='1750px',canvas_height='500px',theme='dark') -> HeatMap:
    df = pd.read_csv(file_path)
    df_corr = df.corr(method="pearson")

    index = df_corr.index.tolist()
    columns = df_corr.columns.tolist()

    data = []
    for i in range(len(index)):
        for j in range(len(columns)):
            data.append([i, j, round(df_corr.iloc[i, j], 2)])

    heatmap = (
        HeatMap(
            init_opts=opts.InitOpts(
                theme=theme, width=canvas_width, height=canvas_height
            )
        )
        .add_xaxis(columns)
        .add_yaxis(
            "",
            index,
            data,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="关联矩阵热力图", pos_left="center", pos_top=20),
            visualmap_opts=opts.VisualMapOpts(
                min_=df_corr.min().min(),
                max_=df_corr.max().max(),
                is_calculable=True,
                orient="vertical",
                pos_top="bottom",
                pos_left=15,
            ),
            # xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-20)),
            legend_opts=opts.LegendOpts(pos_top="bottom"),
        )
    )

    return heatmap


def plot_box(file_path,theme='dark') -> Boxplot:
    df = pd.read_csv(file_path).drop(columns="DATATIME")

    # 构造数据
    data1 = df["WINDSPEED"].tolist()
    data2 = df["PREPOWER"].tolist()
    data3 = df["WINDDIRECTION"].tolist()
    data4 = df["TEMPERATURE"].tolist()
    data5 = df["HUMIDITY"].tolist()
    data6 = df["PRESSURE"].tolist()
    data7 = df["ROUND(A.WS,1)"].tolist()
    data8 = df["ROUND(A.POWER,0)"].tolist()
    data9 = df["YD15"].tolist()

    # 创建盒须图对象
    boxplot1 = (
        Boxplot(init_opts=opts.InitOpts(theme=theme))
        .add_xaxis(["WINDSPEED"])
        .add_yaxis("", Boxplot().prepare_data([data1]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    boxplot2 = (
        Boxplot(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(["PREPOWER"])
        .add_yaxis("", Boxplot().prepare_data([data2]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    boxplot3 = (
        Boxplot(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(["WINDDIRECTION"])
        .add_yaxis("", Boxplot().prepare_data([data3]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    boxplot4 = (
        Boxplot(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(["TEMPERATURE"])
        .add_yaxis("", Boxplot().prepare_data([data4]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    boxplot5 = (
        Boxplot(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(["HUMIDITY"])
        .add_yaxis("", Boxplot().prepare_data([data5]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    boxplot6 = (
        Boxplot(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(["PRESSURE"])
        .add_yaxis("", Boxplot().prepare_data([data6]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    boxplot7 = (
        Boxplot(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(["ROUND(A.WS,1)"])
        .add_yaxis("", Boxplot().prepare_data([data7]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    boxplot8 = (
        Boxplot(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(["ROUND(A.POWER,0)"])
        .add_yaxis("", Boxplot().prepare_data([data8]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    boxplot9 = (
        Boxplot(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(["YD15"])
        .add_yaxis("", Boxplot().prepare_data([data9]), box_width="40%")
        .set_global_opts(title_opts=opts.TitleOpts(title="盒须图"))
    )

    # 创建网格对象
    grid = (
        Grid(
            init_opts=opts.InitOpts(
                width="1600px", height="400px", theme=ThemeType.DARK
            )
        )
        .add(boxplot1, grid_opts=opts.GridOpts(pos_left="2%", pos_right="92%"))
        .add(boxplot2, grid_opts=opts.GridOpts(pos_left="13.5%", pos_right="80.5%"))
        .add(boxplot3, grid_opts=opts.GridOpts(pos_left="23.5%", pos_right="70.5%"))
        .add(boxplot4, grid_opts=opts.GridOpts(pos_left="33.5%", pos_right="60.5%"))
        .add(boxplot5, grid_opts=opts.GridOpts(pos_left="43.5%", pos_right="50.5%"))
        .add(boxplot6, grid_opts=opts.GridOpts(pos_left="53.5%", pos_right="40.5%"))
        .add(boxplot7, grid_opts=opts.GridOpts(pos_left="63.5%", pos_right="30.5%"))
        .add(boxplot8, grid_opts=opts.GridOpts(pos_left="73.5%", pos_right="20.5%"))
        .add(boxplot9, grid_opts=opts.GridOpts(pos_left="83.5%", pos_right="10.5%"))
    )
    return grid


import pyecharts.options as opts
from pyecharts.charts import Scatter


def plot_ts_scatter_2d(
    df,
    col1,
    col2,
    *,
    theme="dark",#主题，light,dark,chalk,essos,infographic,macarons等，详见https://pyecharts.org/#/zh-cn/themes
    canvas_width="1250px",#画布宽，str
    canvas_height="600px",#画布高
    border_width=0.8,#点轮廓线宽度
    border_color="white",#轮廓线颜色，
    title_top=20,#标题距顶部距离
    s_color="#7AC9FF",#起始颜色，rgb表示或者blue等
    e_color="#0459FF",#最高颜色
    symbol_size=10,#点大小
) -> Scatter:
    if type(df) == str:
        df = pd.read_csv(df)
    scatter = (
        Scatter(
            init_opts=opts.InitOpts(
                theme=theme,
                width=canvas_width,
                height=canvas_height,
                animation_opts=opts.AnimationOpts(
                    animation=False, animation_duration=0, animation_easing="cubicInOut"
                ),
            )
        )
        .add_xaxis(xaxis_data=[float(i) for i in df[col1]])
        .add_yaxis(
            series_name=str(col2),
            y_axis=[float(i) for i in df[col2]],
            symbol_size=symbol_size,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_series_opts(
            itemstyle_opts=opts.ItemStyleOpts(
                border_color=border_color, border_width=border_width, opacity=1
            )
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                name=col1,
                type_="value",
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            yaxis_opts=opts.AxisOpts(
                name=col2,
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            visualmap_opts=opts.VisualMapOpts(
                max_=df[col2].max(), range_color=[s_color, e_color], pos_left=15
            ),
            title_opts=opts.TitleOpts(
                title=f"{col1}与{col2}的关系图", pos_left="center", pos_top=title_top
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    return scatter


def plot_ts_scatter_3d(
    df,
    col1,
    col2,
    col3,
    *,
    canvas_width="1250px",
    canvas_height="600px",#画布大小
    grid_width=200,
    grid_height=100,
    grid_depth=200,#坐标系大小
    border_width=0.05,#轮廓线宽度
    border_color="white",#轮廓线颜色
    title_top=50,#标题距顶部距离
    theme='dark',#主题
    is_rotate=False#是否自动旋转
) -> Scatter3D:
    if type(df) == str:
        df = pd.read_csv(df)
    scatter3d = (
        Scatter3D(
            init_opts=opts.InitOpts(
                theme=ThemeType.DARK, width=canvas_width, height=canvas_height
            )
        )
        .add(
            col3,
            data=[
                [float(df[col1][i]), float(df[col2][i]), float(df[col3][i])]
                for i in range(len(df))
            ],
            xaxis3d_opts=opts.Axis3DOpts(
                name=col1,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            yaxis3d_opts=opts.Axis3DOpts(
                name=col2,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            zaxis3d_opts=opts.Axis3DOpts(
                name=col3,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white", margin=10),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            grid3d_opts=opts.Grid3DOpts(
                width=grid_width, height=grid_height, depth=grid_depth,is_rotate=is_rotate
            ),
        )
        .set_series_opts(
            itemstyle_opts=opts.ItemStyleOpts(
                border_width=border_width, border_color=border_color
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{col3}与{col1},{col2}的关系图", pos_left="center", pos_top=title_top
            ),
            visualmap_opts=opts.VisualMapOpts(
                max_=df[col3].max(), range_color=Faker.visual_color, pos_left=15
            ),
            legend_opts=opts.LegendOpts(border_width=0, is_show=False),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                    restore=opts.ToolBoxFeatureRestoreOpts(is_show=False),
                ),
            ),
        )
    )
    return scatter3d


# =========================异常检测之前进行填充的函数============================
def KNN_Impute(df):
    # 异常检测之前的KNN填充
    numeric_columns = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
        "ROUND(A.POWER,0)",
        "YD15",
    ]
    imputer = KNNImputer(n_neighbors=500, weights="distance")
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


def KNN_RF_Impute(df):
    # 异常检测之前的KNN+RF混合填充
    features = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
    ]
    imputer = KNNImputer(n_neighbors=500, weights="distance")
    df[features] = imputer.fit_transform(df[features])
    targets = ["ROUND(A.POWER,0)", "YD15"]
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


# =========================异常检测可选择的函数============================
def IF_Detection(df):
    # 孤立森林检测
    df = df.set_index("DATATIME")
    IF = IsolationForest(contamination=0.25)  # 设置异常值比例
    numeric_columns = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
        "ROUND(A.POWER,0)",
        "YD15",
    ]
    IF.fit(df[numeric_columns])
    outliers = IF.predict(df[numeric_columns])
    nan_index = outliers == -1
    df.loc[nan_index, ["ROUND(A.POWER,0)", "YD15"]] = np.nan
    return df


def LOF_Detection(df):
    # LOF检测（基于密度度量）
    from sklearn.neighbors import LocalOutlierFactor

    numeric_columns = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
        "ROUND(A.POWER,0)",
        "YD15",
    ]
    df = df.set_index("DATATIME")
    LOF = LocalOutlierFactor(n_neighbors=20, contamination=0.25)  # 设置邻居数和异常值比例
    outlier_scores = LOF.fit_predict(df[numeric_columns])
    nan_index = outlier_scores == -1
    df.loc[nan_index, ["ROUND(A.POWER,0)", "YD15"]] = np.nan
    return df


def EllipticEnvelope_Detection(df):
    # 椭圆包络检测（基于鲁棒协方差估计的异常值检测）
    from sklearn.covariance import EllipticEnvelope

    numeric_columns = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
        "ROUND(A.POWER,0)",
        "YD15",
    ]
    df = df.set_index("DATATIME")
    clf = EllipticEnvelope(contamination=0.25)  # 设置异常值比例
    clf.fit(df[numeric_columns])
    outliers = clf.predict(df[numeric_columns])
    nan_index = outliers == -1
    df.loc[nan_index, ["ROUND(A.POWER,0)", "YD15"]] = np.nan
    return df


def OneClassSVM_Detection(df):
    # One-Class SVM检测（基于支持向量机的异常值检测）
    from sklearn.svm import OneClassSVM

    numeric_columns = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
        "ROUND(A.POWER,0)",
        "YD15",
    ]
    df = df.set_index("DATATIME")
    clf = OneClassSVM(nu=0.25)  # 设置异常值比例
    clf.fit(df[numeric_columns])
    outliers = clf.predict(df[numeric_columns])
    nan_index = outliers == -1
    df.loc[nan_index, ["ROUND(A.POWER,0)", "YD15"]] = np.nan
    return df


# =========================异常检测之后进行填充的函数============================
def KNN_Impute_after(df):
    # 异常检测之后的KNN填充
    numeric_columns = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
        "ROUND(A.POWER,0)",
        "YD15",
    ]
    imputer = KNNImputer(n_neighbors=100, weights="distance")
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


def RF_Impute_after(df):
    # 异常检测之后的随机森林填充
    df["ROUND(A.WS,1)"] = df["ROUND(A.WS,1)"].interpolate(
        method="linear", limit_direction="both"
    )
    features = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
    ]
    targets = ["ROUND(A.POWER,0)", "YD15"]

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


def data_preprocess(
    file_path,
    before_impute_method,
    detection,
    after_impute_method,
    *,
    canvas_width="1250px",
    canvas_height="600px",
    grid_width=200,#坐标系的大小
    grid_height=100,
    grid_depth=200,
    border_width=0.05,
    border_color="white",
    title_top=50,
    theme='dark',#主题
    normal_color='66ccff',#正常点的颜色
    outliers_color='fd1105',#异常点的颜色
    impute_color='ffff00',#填充点的颜色
    is_rotate=False#是否自动旋转
):
    """数据预处理
    1. 根据时间排序，并去除重复样本
    2. 处理离群样本点
    3. 处理一天内YD15持续不变的样本点
    4. 填充缺失值
    5. 纠正异常值
    """

    # 初步处理
    df = pd.read_csv(file_path)
    df = df.sort_values(by="DATATIME", ascending=True)
    df = df.drop_duplicates(subset="DATATIME", keep="first")
    if df['YD15'].isnull().all() == True:
        df['YD15'] = df["ROUND(A.POWER,0)"]
    df.reset_index(drop=True, inplace=True)
    columns_to_keep = [
        "TurbID",
        "DATATIME",
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
        "ROUND(A.POWER,0)",
        "YD15",
    ]
    numeric_columns = [
        "WINDSPEED",
        "PREPOWER",
        "WINDDIRECTION",
        "TEMPERATURE",
        "HUMIDITY",
        "PRESSURE",
        "ROUND(A.WS,1)",
        "ROUND(A.POWER,0)",
        "YD15",
    ]
    df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])

    # 找出离群点并设为空值
    out_cols = ["ROUND(A.WS,1)", "ROUND(A.POWER,0)", "YD15"]
    for col in out_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        outliers_index = df[outliers].index
        df.loc[outliers_index, col] = np.nan

    # 找出一天内持续不变的YD15设为空值
    # 将时间列转换为 datetime 类型
    df["DATATIME"] = pd.to_datetime(df["DATATIME"])
    df["DATE"] = df["DATATIME"].dt.date
    grouped = df.groupby(["DATE"])
    unique_counts = grouped["YD15"].nunique()
    stale_rows = unique_counts[unique_counts == 1].index.tolist()
    stale_indices = []
    for date in stale_rows:
        indices = df.index[(df["DATE"] == date)]
        stale_indices.extend(indices.tolist())
    df.loc[stale_indices, "YD15"] = np.nan
    df.drop(columns=["DATE"], inplace=True)

    # 特殊异常值处理
    df.loc[df["ROUND(A.WS,1)"] < 0, "ROUND(A.WS,1)"] = np.nan
    df.loc[df["ROUND(A.WS,1)"] > 1e6, "ROUND(A.WS,1)"] = np.nan
    columns = ["ROUND(A.POWER,0)", "YD15"]
    for col in columns:
        df.loc[(df[col] > 1e6) | (df[col] < -1e6), col] = np.nan
        df.loc[(df["ROUND(A.WS,1)"] == 0) & (df[col] > 0), col] = 0
        df.loc[(df["ROUND(A.WS,1)"] > 20) & (df[col] != 0), col] = 0
        df.loc[(df["ROUND(A.WS,1)"] > 5) & (df[col] == 0), col] = np.nan

    # 检测异常值之前填充
    if before_impute_method == "KNN":
        df = KNN_Impute(df)
    elif before_impute_method == "KNN_RF":
        df = KNN_RF_Impute(df)

    # 检测异常值
    if detection == "IF":
        df = IF_Detection(df)
    elif detection == "LOF":
        df = LOF_Detection(df)
    elif detection == "EllipticEnvelope":
        df = EllipticEnvelope_Detection(df)
    elif detection == "SVM":
        df = OneClassSVM_Detection(df)
    col1, col2, col3 = "ROUND(A.WS,1)", "ROUND(A.POWER,0)", "YD15"
    nan_index = [
        i
        for i in df.loc[
            (df["ROUND(A.WS,1)"].isna() == True)
            | (df["ROUND(A.POWER,0)"].isna() == True)
            | (df["YD15"].isna() == True),
            "YD15",
        ].index
    ]
    data_normal = []
    for i in range(len(df)):
        if i not in nan_index:
            data_normal.append(
                [float(df[col1][i]), float(df[col2][i]), float(df[col3][i])]
            )
    scatter3d_outliers = (
        Scatter3D(init_opts=opts.InitOpts(theme=theme,width=canvas_width,height=canvas_height))
        .add(
            "正常值",
            data=data_normal,
            xaxis3d_opts=opts.Axis3DOpts(
                name=col1,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            yaxis3d_opts=opts.Axis3DOpts(
                name=col2,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            zaxis3d_opts=opts.Axis3DOpts(
                name=col3,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            grid3d_opts=opts.Grid3DOpts(width=grid_width, depth=grid_depth,height=grid_height,is_rotate=is_rotate),
            itemstyle_opts=opts.ItemStyleOpts(
                border_color=border_color, border_width=border_width,color=normal_color
            )
        )
        .add(
            "异常值",
            data=[
                [float(df[col1][i]), float(df[col2][i]), float(df[col3][i])]
                for i in nan_index
            ],
            xaxis3d_opts=opts.Axis3DOpts(
                name=col1,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            yaxis3d_opts=opts.Axis3DOpts(
                name=col2,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            zaxis3d_opts=opts.Axis3DOpts(
                name=col3,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            grid3d_opts=opts.Grid3DOpts(width=grid_width, depth=grid_depth,height=grid_height,is_rotate=is_rotate),
            itemstyle_opts=opts.ItemStyleOpts(
                border_color=border_color, border_width=border_width,color=outliers_color
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{col3}与{col1},{col2}的关系图", pos_left="center", pos_top=title_top
            ),
            legend_opts=opts.LegendOpts(border_width=0),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                    restore=opts.ToolBoxFeatureRestoreOpts(is_show=False),
                ),
            )
        )
    )

    # 检测异常值之后填充
    if after_impute_method == "KNN":
        df = KNN_Impute_after(df)
    elif after_impute_method == "RF":
        df = RF_Impute_after(df)

    scatter3d_impute = (
             Scatter3D(init_opts=opts.InitOpts(theme=theme,width=canvas_width,height=canvas_height))
        .add(
            "原值",
            data=data_normal,
            xaxis3d_opts=opts.Axis3DOpts(
                name=col1,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            yaxis3d_opts=opts.Axis3DOpts(
                name=col2,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            zaxis3d_opts=opts.Axis3DOpts(
                name=col3,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            grid3d_opts=opts.Grid3DOpts(width=grid_width, depth=grid_depth,height=grid_height,is_rotate=is_rotate),
            itemstyle_opts=opts.ItemStyleOpts(
                border_color=border_color, border_width=border_width,color=normal_color
            )
        )
        .add(
            "填充值",
            data=[
                [float(df[col1][i]), float(df[col2][i]), float(df[col3][i])]
                for i in nan_index
            ],
            xaxis3d_opts=opts.Axis3DOpts(
                name=col1,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            yaxis3d_opts=opts.Axis3DOpts(
                name=col2,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            zaxis3d_opts=opts.Axis3DOpts(
                name=col3,
                type_="value",
                axislabel_opts=opts.LabelOpts(color="white"),
                textstyle_opts=opts.TextStyleOpts(color="white"),
            ),
            grid3d_opts=opts.Grid3DOpts(width=grid_width, depth=grid_depth,height=grid_height,is_rotate=is_rotate),
            itemstyle_opts=opts.ItemStyleOpts(
                border_color=border_color, border_width=border_width,color=impute_color
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{col3}与{col1},{col2}的关系图", pos_left="center", pos_top=title_top
            ),
            legend_opts=opts.LegendOpts(border_width=0),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                    restore=opts.ToolBoxFeatureRestoreOpts(is_show=False),
                ),
            )
        )
    )

    save_path = (
        f"./static/{before_impute_method}_{detection}_{after_impute_method}"
        + file_path.split("/")[len(file_path.split("/")) - 1]
    )
    df.to_csv(save_path)
    return jsonify({"save_path":save_path,"outliers": json.loads(scatter3d_outliers.dump_options_with_quotes()),"impute":json.loads(scatter3d_impute.dump_options_with_quotes())})