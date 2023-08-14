import random
from datetime import datetime, timedelta


def now_alarm_data():
    """
    设备信息以及实时数据。
    返回两个字典，第一个是报警信息，第二个是实时数据。实时数据相比PPT中的临时加了一行（电流）
    """
    rand_idx = random.randint(0, 19)
    id_list = list(range(11, 31))
    type_list = ["LYP-W" + str(f"{x:03d}") for x in list(range(11, 31))]
    local_list = [
        "广东省/广州市/天河区",
        "湖北省/武汉市/江岸区",
        "浙江省/杭州市/西湖区",
        "山东省/青岛市/崂山区",
        "河南省/郑州市/中原区",
        "湖南省/长沙市/岳麓区",
        "四川省/成都市/武侯区",
        "北京市/北京市/朝阳区",
        "上海市/上海市/黄浦区",
        "广东省/深圳市/南山区",
        "河北省/石家庄市/长安区",
        "江苏省/南京市/秦淮区",
        "辽宁省/大连市/甘井子区",
        "陕西省/西安市/雁塔区",
        "山西省/太原市/小店区",
        "河北省/唐山市/路南区",
        "湖南省/株洲市/天元区",
        "浙江省/温州市/鹿城区",
        "福建省/厦门市/思明区",
        "广东省/珠海市/香洲区"
    ]
    state_list = ['良好', '正常', '异常', '故障']
    windspeed_range = (0, 20)
    winddir_range = (0, 180)
    humidity_range = (80, 98)
    pressure_range = (980, 1020)
    rospeed_range = (0, 8000)
    temper_range = (0, 500)
    voltage_range = (0, 5000)
    elecurrent_range = (0, 500)
    power_range = (0, 100000)
    unit_list = ['m/s', '°', '%', 'Pa', 'r/s', '℃', 'V', 'A', 'W']
    warning = {
        'id': id_list[rand_idx],
        'type': type_list[rand_idx],
        'local': local_list[rand_idx],
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'state': random.choice(state_list),
    }
    cur_info = {
        'fs': (round(random.uniform(*windspeed_range), 3), unit_list[0]),
        'fx': (round(random.uniform(*winddir_range), 3), unit_list[1]),
        'sd': (round(random.uniform(*humidity_range), 3), unit_list[2]),
        'yq': (round(random.uniform(*pressure_range), 3), unit_list[3]),
        'zs': (round(random.uniform(*rospeed_range), 3), unit_list[4]),
        'wd': (round(random.uniform(*temper_range), 3), unit_list[5]),
        'dy': (round(random.uniform(*voltage_range), 3), unit_list[6]),
        'dl': (round(random.uniform(*elecurrent_range), 3), unit_list[7]),
        'gl': (round(random.uniform(*power_range), 3), unit_list[8])
    }
    return (warning, cur_info)


# 20个中国地名
locations = ["重庆/渝中区", "上海/黄浦区", "广东/广州市/天河区", "北京/朝阳区", "浙江/杭州市/西湖区",
             "江苏/南京市/鼓楼区", "四川/成都市/武侯区", "山东/青岛市/市北区", "湖北/武汉市/江岸区",
             "辽宁/沈阳市/和平区", "陕西/西安市/雁塔区", "福建/福州市/鼓楼区", "湖南/长沙市/岳麓区",
             "河北/石家庄市/长安区", "河南/郑州市/中原区", "吉林/长春市/南关区", "黑龙江/哈尔滨市/道里区",
             "山西/太原市/杏花岭区", "甘肃/兰州市/城关区", "云南/昆明市/五华区"]

alarm_types = ["温度过高", "风速异常", "通信故障", "设备故障", "电流异常"]
alarm_levels = ["紧急", "严重", "一般"]
handling_results = ["已解决", "处理中", "待处理"]
handling_notes = [
    "已检查现场设备状态，确认为误报。",
    "已进行紧急维护，更换故障传感器。",
    "正在观察设备运行情况，暂不做处理。",
    "已通知维护人员前往现场处理。",
    "设备电缆连接松动，已重新插拔连接。",
    "已调整风机运行参数，降低风速以解决报警。",
    "设备工作温度正常，报警可能由温度测量误差引起。",
    "通信故障已修复，设备重新连接网络。",
    "部分电缆受损，已更换电缆以恢复设备运行。",
    "报警源于电网异常，等待电网稳定后报警自动消除。"
]


def hist_alarm_data():
    """
    历史报警。
    返回一个包括20个字典的列表，随机模拟的历史20天的报警信息
    """
    data = []
    alarm_time = datetime.now() - timedelta(days=random.randint(1, 30))
    for _ in range(20):
        alarm_time -= timedelta(days=random.randint(7, 60))
        alarm_type = random.choice(alarm_types)
        alarm_level = random.choice(alarm_levels)
        alarm_description = f"发生{alarm_type}报警"
        alarm_handler = f"处理人员{random.randint(1, 5)}"
        handled_time = alarm_time + timedelta(hours=random.randint(1, 48))
        handling_result = random.choice(handling_results)
        handling_note = random.choice(handling_notes)

        alarm_data = {
            "alarm_time": alarm_time.strftime("%Y-%m-%d %H:%M:%S"),
            "alarm_type": alarm_type,
            "alarm_level": alarm_level,
            "alarm_description": alarm_description,
            # "报警位置": location,
            "process_name": alarm_handler,
            "process_time": handled_time.strftime("%Y-%m-%d %H:%M:%S"),
            "process_result": handling_result,
            "process_way": handling_note,
            # "报警状态": alarm_status
        }
        data.append(alarm_data)
    return data


def hist_data():
    """
    历史数据。
    返回一个包括20个字典的列表，包括历史20天的详细数据信息，字典的最后一个键值是日期
    """
    current_date = datetime.now()
    date_list = []
    for _ in range(20):
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date -= timedelta(days=1)
    data_list = []
    for date in date_list:
        data = now_alarm_data()[1]
        data['time'] = ('', date)
        values = [str(item[0]) + item[1] for item in data.values()]
        keys = ['wind_speed','wind_dir','humidity','pressure','rot_speed','temperature','voltage','current','power','time']
        data_list.append(dict(zip(keys, values)))
    return data_list