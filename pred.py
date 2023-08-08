from paddle import inference as infer
import pandas as pd
import numpy as np
import paddle
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pyecharts import options as opts
from pyecharts.charts import Bar, Scatter
from pyecharts.components import Table
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot


def feature_engineer(df):
    """特征工程：时间戳特征"""
    # 时间戳特征
    df['month'] = df.DATATIME.apply(lambda row: row.month, 1)
    df['day'] = df.DATATIME.apply(lambda row: row.day, 1)
    df['hour'] = df.DATATIME.apply(lambda row: row.hour, 1)
    df['minute'] = df.DATATIME.apply(lambda row: row.minute, 1)

    # TODO 挖掘更多特征：差分序列、同时刻风场/邻近风机的特征均值/标准差等

    # # 差分序列
    df['YD15_diff'] = df['YD15'].diff()
    df["apow"] = df["ROUND(A.POWER,0)"].diff()
    df['POWER_diff'] = df['PREPOWER'].diff()
    df['RWS_diff'] = df['ROUND(A.WS,1)'].diff()
    df['wp'] = df["WINDSPEED"].diff()

    # 邻近风机的特征均值/标准差
    neighboring_features = ['PREPOWER', 'ROUND(A.POWER,0)', 'YD15']
    df['neighboring_mean'] = df[neighboring_features].mean(axis=1)  # 邻近风机特征均值
    df['neighboring_std'] = df[neighboring_features].std(axis=1)  # 邻近风机特征标准差

    df["yd2"] = df["YD15"] * df["YD15"]
    df["yd3"] = df["YD15"] * df["YD15"] * df["YD15"]
    df['r12'] = df["ROUND(A.WS,1)"] * df["ROUND(A.WS,1)"]
    df['r13'] = df["ROUND(A.WS,1)"] * df["ROUND(A.WS,1)"] * df["ROUND(A.WS,1)"]
    df['apow3'] = df["ROUND(A.POWER,0)"] * df["ROUND(A.POWER,0)"]
    df['apow2'] = df["ROUND(A.POWER,0)"] * df["ROUND(A.POWER,0)"] * df["ROUND(A.POWER,0)"]
    df['wp2'] = df["WINDSPEED"] * df["WINDSPEED"]
    df['wp3'] = df["WINDSPEED"] * df["WINDSPEED"] * df["WINDSPEED"]

    return df


def plot_frequency_histogram(data, bins=10):
    # 统计频率
    counts, edges = np.histogram(data, bins=bins)
    # 计算频率密度
    frequencies = counts / len(data)
    # 获取直方图的中心点
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

    # 绘制直方图
    bar = (
        Bar()
            .add_xaxis([f"{edge:.2f}" for edge in edges])
            .add_yaxis("频数", frequencies.tolist(), category_gap="0%")
            .set_global_opts(
            title_opts=opts.TitleOpts(title="误差分布直方图"),
            xaxis_opts=opts.AxisOpts(name="误差区间"),
            yaxis_opts=opts.AxisOpts(name="频数"),
        )
    )

    return bar


def calc_acc(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return 1 - rmse/201000


def analysis_statement(accuracy, mean_squared_error, mean_absolute_error, pearson_correlation, r2):
    # 根据不同指标值生成不同的误差分析语句
    if accuracy >= 0.85:
        analysis_statement = "我们的模型预测准确率非常高，达到了{:.2f}，表现非常出色。".format(accuracy)
    elif accuracy >= 0.7:
        analysis_statement = "我们的模型预测准确率较高，达到了{:.2f}，预测结果相对可靠。".format(accuracy)
    else:
        analysis_statement = "我们的模型预测准确率较低，仅有{:.2f}，在实际应用中需要谨慎考虑。".format(accuracy)
    analysis_statement+='\n'

    if mean_squared_error <= 1000:
        analysis_statement += "均方误差较小，为{:.2e}，说明预测结果与实际观测值之间的误差较小。".format(mean_squared_error)
    else:
        analysis_statement += "均方误差较大，为{:.2e}，在预测过程中存在较大误差。".format(mean_squared_error)
    analysis_statement+='\n'

    if mean_absolute_error <= 50:
        analysis_statement += "平均绝对误差较小，为{:.2e}，说明预测结果相对准确。".format(mean_absolute_error)
    else:
        analysis_statement += "平均绝对误差较大，为{:.2e}，需要进一步优化模型。".format(mean_absolute_error)
    analysis_statement+='\n'

    if pearson_correlation >= 0.8:
        analysis_statement += "皮尔逊相关系数较高，为{:.2f}，预测结果与实际观测值具有较强的线性相关性。".format(pearson_correlation)
    else:
        analysis_statement += "皮尔逊相关系数较低，为{:.2f}，预测结果与实际观测值之间的线性关系较弱。".format(pearson_correlation)
    analysis_statement+='\n'

    if r2 >= 0.8:
        analysis_statement += "拟合优度较高，为{:.2f}，说明模型在训练数据上表现出色。".format(r2)
    else:
        analysis_statement += "拟合优度较低，为{:.2f}，模型在训练数据上的拟合程度较差，改进空间较大。".format(r2)

    return analysis_statement


def error_analysis(true_power,predicted_power):
    # 计算误差
    errors = np.array([true_power - predicted_power])
    # 计算准确率
    acc = calc_acc(true_power, predicted_power)
    # 计算均方误差
    mse = mean_squared_error(true_power, predicted_power)
    # 计算均方根误差
    rmse = np.sqrt(mse)
    # 计算平均绝对误差
    mae = mean_absolute_error(true_power, predicted_power)

    # 计算皮尔逊相关系数
    correlation_coefficient = np.corrcoef(true_power, predicted_power)[0, 1]
    # 判断相关性强弱
    correlation_strength = ''
    if abs(correlation_coefficient) >= 0.8:
        correlation_strength = '强'#强
    elif abs(correlation_coefficient) >= 0.6:
        correlation_strength = '中等'#中等
    else:
        correlation_strength = '弱'#弱

    # 计算拟合优度
    r_squared = r2_score(true_power, predicted_power)
    # 判断拟合优度
    fit_quality = ''
    if r_squared >= 0.8:
        fit_quality = '好'#好
    elif r_squared >= 0.6:
        fit_quality = '一般'#一般
    else:
        fit_quality = '差'#差
    print(errors)
    # 绘制误差分布图
    bar = plot_frequency_histogram(errors)

    # 绘制残差图
    scatter = (
        Scatter()
        .add_xaxis(predicted_power.to_list())
        .add_yaxis("残差", errors.tolist()[0], symbol_size=10)
         .set_series_opts(
            itemstyle_opts=opts.ItemStyleOpts(
                border_color='white', border_width=0.8, opacity=1
            )
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                name="所有的点",
                type_="value",
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            yaxis_opts=opts.AxisOpts(
                name="残差值",
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            visualmap_opts=opts.VisualMapOpts(
                max_=max(errors.tolist()[0]), range_color=["#7AC9FF","#0459FF"], pos_left=15
            ),
            title_opts=opts.TitleOpts(
                title=f"残差图", pos_left="center", pos_top=20
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    table_data=pd.DataFrame({'zb':["预测准确率","均方误差","均方根误差","平均绝对误差","皮尔逊相关系数","拟合优度"],
                             'sz':[f"{100*acc:.2f}%",f"{mse:.2f}",f"{rmse:.4f}",f"{mae:.2f}",f"{correlation_coefficient:.4f}",f"{r_squared:.2f}"],
                             'pd':['无','无','无','无',f"{correlation_strength}",f"{fit_quality}"]})
    # 绘制表格
    return bar.dump_options_with_quotes(),scatter.dump_options_with_quotes(),table_data.to_json(double_precision=6), analysis_statement(acc,mse,mae,correlation_coefficient,r_squared)


def pred(file, tid, start, input, output):
    data = pd.read_csv(file)
    data['DATATIME'] = pd.to_datetime(data['DATATIME'])
    # print(data['DATATIME'][0].month)
    data = feature_engineer(data)
    end = data.loc[data['DATATIME'] == start].index
    begin = end - 24 * input * 4
    use_cols = ['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY',
                'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15',
                'month', 'day', 'hour', 'minute',
                'YD15_diff', 'apow', 'POWER_diff', 'RWS_diff', 'wp', 'neighboring_mean', 'neighboring_std', 'yd2',
                'yd3', 'r12', 'r13', 'apow3', 'apow2', 'wp2', 'wp3']
    future_use_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'wp2', 'wp3', 'wp']
    fend = end + 24 * 4 * output
    input1 = data[begin[0]:end[0]][use_cols].to_numpy()
    input2 = data[end[0] + 1:fend[0] + 1][future_use_cols].to_numpy()
    scaler1 = pickle.load(open('./wind/scaler/scaler{}_1.pkl'.format(tid), 'rb'))
    input1 = scaler1.transform(input1)
    scaler2 = pickle.load(open('./wind/scaler/scaler{}_2.pkl'.format(tid), 'rb'))
    input2 = scaler2.transform(input2)

    input1 = paddle.to_tensor(input1).astype('float32')
    input2 = paddle.to_tensor(input2).astype('float32')
    input1 = input1.reshape((1, 24 * input * 4, 28)).numpy()
    input2 = input2.reshape((1, 24 * 4 * output, 8)).numpy()
    config = infer.Config(f"./wind/model/in{input}_out{output}/model_checkpoint_windid_{tid}.pdmodel",
                          f"./wind/model/in{input}_out{output}/model_checkpoint_windid_{tid}.pdiparams")
    config.disable_gpu()
    config.enable_mkldnn()
    config.disable_glog_info()

    predictor = infer.create_predictor(config)
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.copy_from_cpu(input1)
    input_tensor = predictor.get_input_handle(input_names[1])
    input_tensor.copy_from_cpu(input2)
    predictor.run()
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[1])
    output_data = output_tensor.copy_to_cpu().tolist()[0]
    time_stamp = data[end[0] + 1:fend[0] + 1]['DATATIME']
    round_power = data[end[0] + 1:fend[0] + 1]['YD15'].to_list()
    error_data = [(rp - pp) for rp, pp in zip(round_power,output_data)]
    time_list = time_stamp.dt.strftime(('%Y-%m-%d %H:%M:%S')).to_list()
    # pred_df = pd.Series(output_data[0], name='YD15')
    # true_df = data[end[0] + 1:fend[0] + 1]['YD15']
    # true_df = true_df.reset_index()['YD15']
    # return ([time_list, output_data], error_analysis(true_df, pred_df))
    pred_df=pd.Series(output_data,name='YD15')
    true_df=data[end[0] + 1:fend[0] + 1]['YD15']
    true_df=true_df.reset_index()['YD15']
    return [time_list, round_power, output_data, error_data], error_analysis(true_df,pred_df)