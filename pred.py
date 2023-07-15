from paddle import inference as infer
import pandas as pd
import numpy as np
import paddle
def pred(tid,start,len1):
    data=pd.read_csv(f"n{tid}.csv")
    index=data.loc[data['DATATIME']==start].index
    end=index+24*len1*4
    use_cols =['WINDSPEED', 'PREPOWER',  'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY',
                  'PRESSURE','ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15',
                 'month', 'day', 'hour', 'minute' ,
                 'YD15_diff','apow','POWER_diff','RWS_diff','wp','neighboring_mean','neighboring_std','yd2','yd3','r12','r13','apow3','apow2','wp2','wp3']
    future_use_cols =['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE','wp2','wp3','wp']
    fend=end+24*4
    input1=data[index[0]:end[0]][use_cols].to_numpy()
    input1=paddle.to_tensor(input1).astype('float32')
    input1=input1.reshape((1,24*len1*4,28)).numpy()
    input2=data[end[0]+1:fend[0]+1][future_use_cols].to_numpy()
    input2=paddle.to_tensor(input2).astype('float32')
    input2=input2.reshape((1,24*4,8)).numpy()
    config = infer.Config(f"./wind/model/hhh{len1}.pdmodel", f"./wind/model/hhh{tid}_{len1}.pdiparams")
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
    output_tensor = predictor.get_output_handle(output_names[0])
    output_data = output_tensor.copy_to_cpu()
    return output_data[0].tolist()