import csv
import json

import requests
from flask import Flask, request, jsonify
from pred import pred
import data_analysis_process
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from urllib.parse import urljoin
import weather

app = Flask(__name__)

HOSTNAME = "127.0.0.1"
PORT = 3306
USERNAME = "root"
PASSWORD = "root"
DATABASE = "windpowerforecast"

app.config[
    'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"

db = SQLAlchemy(app)


# model文件
class WindFarmUrl(db.Model):
    __tablename__ = 'wind_farm_url'

    id = db.Column(db.Integer, primary_key=True)
    farm_id = db.Column(db.Integer, unique=True, nullable=True)
    origin_file_url = db.Column(db.String, nullable=False)
    process_file_url = db.Column(db.String, nullable=False)


@app.route("/weather", methods=["GET"])
def get_whether():
    # weather_now=weather.get_weather_now()
    # weather_future= weather.get_weather_future()
    return json.loads(weather.grid_mutil_yaxis().dump_options_with_quotes())

@app.route("/getFarmData", methods=['POST'])
def get_farm_data():
    info = request.get_json()
    farm_id = info.get('farmId')
    # Query the wind_farm_url table based on the farm_id
    result = db.session.query(WindFarmUrl).filter(WindFarmUrl.farm_id == farm_id).first()
    if result is None:
        return jsonify({"status": False, "message": "No data found for given farm_id"}), 200
    elif result.process_file_url is None:
        return jsonify({"status": False, "message": "process_file is none"})
    else:
        temp_url = result.process_file_url
        full_url = urljoin('http://127.0.0.1:5000', temp_url)

        response = requests.get(full_url)
        response.raise_for_status()  # If the request failed, this will raise a HTTPError

        data = [row for row in csv.DictReader(response.text.splitlines())]
        return jsonify({"status": True, "message": "查询成功", "data": data, "full_url":full_url}), 200


@app.route("/pred", methods=['POST'])
def get_pred():
    info = request.get_json()
    print(info)
    # tid = int(info.get('tid'))
    file = info.get('file')
    tid = int(info.get('tid'))
    start = info.get('start')
    print(start)
    input = int(info.get('input'))
    output = int(info.get('output'))
    List, Tuple = pred(file, tid,start, input, output)
    time_list, true_list, pred_list, error_list = List
    bar, scatter, table_json, statement = Tuple
    return jsonify({'time_list':time_list, 'true_list':true_list,'pred_list':pred_list, 'error_list':error_list,
                    'bar':json.loads(bar), 'scatter':json.loads(scatter), 'table_json':json.loads(table_json),
                    'statement':statement})
    # return jsonify(result.tolist())  # 转换为 list 并 jsonify


@app.route("/miss_info", methods=['POST'])
def get_miss_info():
    info = request.get_json()
    path = info.get('path')
    return data_analysis_process.miss_info_bar(path).dump_options_with_quotes()


@app.route("/descriptive_table", methods=['POST'])
def get_descriptive_table():
    info = request.get_json()
    path = info.get('path')
    table = data_analysis_process.descriptive_analysis(path)
    return jsonify({'table':json.loads(table)})


@app.route("/correlation_analysis", methods=['POST'])
def get_correlation_analysis():
    info = request.get_json()
    path = info.get('path')
    return data_analysis_process.plot_cor_matrix(path).dump_options_with_quotes()


@app.route("/boxplot", methods=['POST'])
def get_boxplot():
    info = request.get_json()
    path = info.get('path')
    return data_analysis_process.plot_box(path).dump_options_with_quotes()


@app.route("/scatter2d", methods=['POST'])
def get_scatter2d():
    info = request.get_json()
    path = info.get('path')
    x = info.get('x')
    y = info.get('y')
    return data_analysis_process.plot_ts_scatter_2d(path, x, y).dump_options_with_quotes()


@app.route("/scatter3d", methods=['POST'])
def get_scatter3d():
    info = request.get_json()
    path = info.get('path')
    x = info.get('x')
    y = info.get('y')
    z = info.get('z')
    return data_analysis_process.plot_ts_scatter_3d(path, x, y, z).dump_options_with_quotes()

@app.route("/save_process_file", methods=['POST'])
def save_process_file():
    info = request.get_json()
    file_path = info.get('file_path')
    farm_id = info.get('farm_id')
    wind_farm = db.session.query(WindFarmUrl).filter(WindFarmUrl.farm_id == farm_id).first()
    if wind_farm is None:
        return jsonify({"status": False, "message": "No data found for given farm_id"}), 200
    else:
        # Update fields
        wind_farm.process_file_url = file_path
        # Commit the changes
        db.session.commit()
        return jsonify({"status": True,"message": "Data updated successfully"}), 200

@app.route("/data_preprocess", methods=['POST'])
def get_data_preprocess():
    """
    进行数据处理
    输入:文件的路径，before_impute_method,detection,after_impute_method
    输出:处理后的文件路径
    """
    info = request.get_json()
    file_path = info.get('file_path')
    before_impute_method = info.get('before_impute_method')
    detection = info.get('detection')
    after_impute_method = info.get('after_impute_method')
    # file_path, before_impute_method, detection, after_impute_method = request.json.split("?")
    return data_analysis_process.data_preprocess(file_path, before_impute_method, detection, after_impute_method)


if __name__ == '__main__':
    app.run()
