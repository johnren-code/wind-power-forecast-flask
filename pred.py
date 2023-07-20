from paddle import inference as infer
import pandas as pd
import numpy as np
import paddle
from sklearn.preprocessing import StandardScaler
import pickle


def pred(tid, start, input, output):
    data = pd.read_csv(f"./wind/ndata/n{tid}.csv")
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
    output_data = output_tensor.copy_to_cpu()
    print(output_data)
    time_stmap = np.array(data[end[0] + 1:fend[0] + 1]['DATATIME'])
    return np.array([time_stmap, output_data])
# pred(11,'2021-11-05 01:30:00',1,1)