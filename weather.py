import random
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line
import json
import pandas as pd

def get_weather_now():
    """生成当前时间下的天气预报信息"""
    # 定义时间和天气数据
    time_slots = ["08:00", "11:00", "14:00", "17:00", "20:00", "23:00", "02:00", "05:00"]
    weather_conditions = ["晴", "多云", "阵雨", "小雨", "阴"]
    # 定义气温、降水、风速、风向、气压、湿度和云量的范围
    temperature_range = (20, 30)
    precipitation_range = (0, 0.3)
    wind_speed_range = (2.6, 3.3)
    wind_directions = ["西北风", "东北风"]
    pressure_range = (944.4, 946.8)
    humidity_range = (73, 96.8)
    cloudiness_range = (80, 100)
    
    weather = random.choice(weather_conditions)
    temperature = round(random.uniform(temperature_range[0], temperature_range[1]), 1) # ℃
    precipitation = round(random.uniform(precipitation_range[0], precipitation_range[1]), 2) # mm
    wind_speed = round(random.uniform(wind_speed_range[0], wind_speed_range[1]), 1) # m/s
    wind_direction = random.choice(wind_directions)
    pressure = round(random.uniform(pressure_range[0], pressure_range[1]), 1) # hPa
    humidity = round(random.uniform(humidity_range[0], humidity_range[1]), 1) # %
    cloudiness = round(random.uniform(cloudiness_range[0], cloudiness_range[1]), 1) # %
    return json.dumps({
        'weather':weather,'temperature':str(temperature)+'℃', 
        'precipitation':str(precipitation)+'mm', 'wind_speed':str(wind_speed)+'m/s', 
        'wind_direction':wind_direction, 'pressure':str(pressure)+'hPa', 
        'humidity':str(humidity)+'%', 'cloudiness':str(cloudiness)+'%',
    })

def get_weather_future():
    """生成未来某一天的8个时间点的天气预报信息，如果需要预测几天，则写个循环，把这个代码执行多次"""
    # 定义时间和天气数据
    time_slots = ["08:00", "11:00", "14:00", "17:00", "20:00", "23:00", "02:00", "05:00"]
    weather_conditions = ["晴", "多云", "阵雨", "小雨", "阴"]
    # 定义气温、降水、风速、风向、气压、湿度和云量的范围
    temperature_range = (20, 30)
    precipitation_range = (0, 0.3)
    wind_speed_range = (2.6, 3.3)
    wind_directions = ["西北风", "东北风","北风","南风","西风","东南风","西南风"]
    pressure_range = (944.4, 946.8)
    humidity_range = (73, 96.8)
    cloudiness_range = (80, 100)
    weather = random.choice(weather_conditions)
    temperature = round(random.uniform(temperature_range[0], temperature_range[1]), 1) # ℃
    precipitation = round(random.uniform(precipitation_range[0], precipitation_range[1]), 2) # mm
    wind_speed = round(random.uniform(wind_speed_range[0], wind_speed_range[1]), 1) # m/s
    wind_direction = random.choice(wind_directions)
    pressure = round(random.uniform(pressure_range[0], pressure_range[1]), 1) # hPa
    humidity = round(random.uniform(humidity_range[0], humidity_range[1]), 1) # %
    cloudiness = round(random.uniform(cloudiness_range[0], cloudiness_range[1]), 1) # %
    res=pd.DataFrame([{
        'time':time_slots[i], 'weather':weather,'temperature':str(temperature)+'℃', 
        'precipitation':str(precipitation)+'mm', 'wind_speed':str(wind_speed)+'m/s', 
        'wind_direction':wind_direction, 'pressure':str(pressure)+'hPa', 
        'humidity':str(humidity)+'%', 'cloudiness':str(cloudiness)+'%',
    } for i in range(len(time_slots))])
    return res.to_json()


import json
import pandas as pd
from datetime import datetime

def get_weather_data(file_path):
    df = pd.read_csv(file_path)
    times = df.iloc[-119*6::6]['DATATIME'].tolist()
    windspeed = df.iloc[-119*6::6]['ROUND(A.WS,1)'].tolist()
    winddirection = df.iloc[-119*6::6]['WINDDIRECTION'].tolist()
    humidity = df.iloc[-119*6::6]['HUMIDITY'].tolist()
    dt_objects = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S") for time in times]
    iso_datetimes = [dt_object.strftime("%Y-%m-%dT%H:%M:%S.000Z") for dt_object in dt_objects]
    dates = [time.split(' ')[0] for time in times]
    dates = sorted(list(set(dates)))

    # 读取json文件内容,返回字典格式
    with open('./weather_data.json','r')as fp:
        dic = json.load(fp)

        data_list = dic['data']
        data_ret_list = []
        for i, data in enumerate(data_list):
            data['time'] = iso_datetimes[i]
            data['windSpeed'] = windspeed[i]
            data['R'] = winddirection[i]
            data['waveHeight'] = humidity[i]
            data_ret_list.append(data)
        dic['data'] = data_ret_list
        
        forecast_list = dic['forecast']
        fore_ret_list = []
        dic['forecast'] = []
        for i, fore in enumerate(forecast_list):
            if i >= len(dates):
                break;
            fore['localDate'] = dates[i]
            fore_ret_list.append(fore)
        dic['forecast'] = fore_ret_list

        json_data = json.dumps(dic)
        
    return json_data


def generate_data_list(new_min, new_max):
    data = np.histogram(np.random.normal(0, 1, 1000), bins=12, density=False)[0].tolist()
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(x - min_value) / (max_value - min_value) * (new_max - new_min) + new_min for x in data]
    return [round(x, 2) for x in normalized_data]


def grid_mutil_yaxis() -> Grid:
    """生成某一个风场的历史天气信息数据"""
    x_data = ["{}月".format(i) for i in range(1, 13)]
    bar = (
        Bar()
        .add_xaxis(x_data)
        .add_yaxis(
            "平均风速",
            generate_data_list(0, 12.5),
            yaxis_index=1,
            color="#5793f3",
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="风速",
                type_="value",
                min_=0,
                max_=15,
                position="right",
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#d14a61")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} m/s"),
            )
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                type_="value",
                name="温度",
                min_=-5,
                max_=35,
                position="left",
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#675bba")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} °C"),
                splitline_opts=opts.SplitLineOpts(
                    is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=1)
                ),
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="气候背景", subtitle="2022年平均风速和平均气温"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        )
    )

    line = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis(
            "平均气温",
            generate_data_list(0, 30),
            yaxis_index=2,
            color="#675bba",
            label_opts=opts.LabelOpts(is_show=False),
        )
    )

    bar.overlap(line)
    return Grid().add(
        bar, opts.GridOpts(pos_left="5%", pos_right="20%", pos_top="17%"), is_control_axis_index=True
    )
