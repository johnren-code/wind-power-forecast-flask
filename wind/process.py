import csv
import os
import re

import mysql.connector

# 连接到数据库
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="root",
    database="windpowerforecast"
)
cursor = conn.cursor()

# 定义你的CSV文件所在的文件夹路径
folder_path = './ndata'

# 循环读取文件夹中的每一个CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 获取不包括.csv后缀的文件名
        numbers = re.findall(r'\d+', filename)
        farm_id = numbers[0]
        with open(os.path.join(folder_path, filename), 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)  # 跳过标题行
            yd15_index = headers.index('YD15')  # 找到 "YD15" 列的索引
            for row in csvreader:
                # 添加farm_id到行的末尾
                selected_data = row[1:yd15_index + 1]
                print(selected_data)
                selected_data.append(farm_id)
                selected_data.append(1)
                cursor.execute('''INSERT INTO wind_farm (
                                    datetime, wind_speed_real, prepower, wind_direction,
                                    temperature, humidity, pressure, round_ws,
                                    round_power, yd15, farm_id, type
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', selected_data)


# 提交更改并关闭数据库连接
conn.commit()
conn.close()
