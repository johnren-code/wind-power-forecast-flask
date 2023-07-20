from flask import Flask, request, jsonify
from pred import pred
import data_analysis_process
app = Flask(__name__)


@app.route("/pred",methods=['POST'])
def get_pred():
    info=request.get_json()
    print(info)
    tid = int(info.get('tid'))
    start = info.get('start')
    print(start)
    input = int(info.get('input'))
    output = int(info.get('output'))
    result = pred(tid, start, input, output)
    result_as_list = [result_item.tolist() for result_item in result]  # 这里确保了所有的 ndarray 都被转换成了列表
    return jsonify(result_as_list)
    # return jsonify(result.tolist())  # 转换为 list 并 jsonify

@app.route("/miss_info",methods=['GET'])
def get_miss_info():
     """
     缺失值统计
     输入:文件的路径
     输出:json格式
          如'{"\\u7f3a\\u5931\\u8ba1\\u6570":{"TurbID":0,"DATATIME":0,"WINDSPEED":0,"PREPOWER":0,"WINDDIRECTION":0,"TEMPERATURE":0,"HUMIDITY":0,"PRESSURE":0,"ROUND(A.WS,1)":0,"ROUND(A.POWER,0)":0,"YD15":0},
          "\\u7f3a\\u5931\\u767e\\u5206\\u6bd4":{"TurbID":0.0,"DATATIME":0.0,"WINDSPEED":0.0,"PREPOWER":0.0,"WINDDIRECTION":0.0,"TEMPERATURE":0.0,"HUMIDITY":0.0,"PRESSURE":0.0,"ROUND(A.WS,1)":0.0,"ROUND(A.POWER,0)":0.0,"YD15":0.0}}'
     """
     path=request.json
     return data_analysis_process.cal_miss_info(path)

@app.route("/descriptive_analysis",methods=['GET'])
def get_descriptive_analysis():
     """
     一些基本统计量
     输入:文件的路径
     输出:json格式
          如：'{"TurbID":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":1.0,"\\u6807\\u51c6\\u5dee":0.0,"\\u6700\\u5c0f\\u503c":1.0,"\\u4e2d\\u4f4d\\u6570":1.0,"\\u6700\\u5927\\u503c":1.0,"\\u65b9\\u5dee":0.0,"\\u5cf0\\u5ea6":0.0,"\\u504f\\u5ea6":0.0,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.0},"WINDSPEED":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":3.4,"\\u6807\\u51c6\\u5dee":0.1,"\\u6700\\u5c0f\\u503c":3.3,"\\u4e2d\\u4f4d\\u6570":3.4,"\\u6700\\u5927\\u503c":3.5,"\\u65b9\\u5dee":0.01,"\\u5cf0\\u5ea6":-3.0,"\\u504f\\u5ea6":0.0,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.0294117647},"PREPOWER":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":10299.2,"\\u6807\\u51c6\\u5dee":1206.2510932638,"\\u6700\\u5c0f\\u503c":8773.0,"\\u4e2d\\u4f4d\\u6570":10300.0,"\\u6700\\u5927\\u503c":11824.0,"\\u65b9\\u5dee":1455041.7000000002,"\\u5cf0\\u5ea6":-1.2037699541,"\\u504f\\u5ea6":-0.002071232,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.1171208534},"WINDDIRECTION":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":122.4,"\\u6807\\u51c6\\u5dee":1.949358869,"\\u6700\\u5c0f\\u503c":120.0,"\\u4e2d\\u4f4d\\u6570":123.0,"\\u6700\\u5927\\u503c":125.0,"\\u65b9\\u5dee":3.8,"\\u5cf0\\u5ea6":-0.8171745152,"\\u504f\\u5ea6":0.080998291,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.0159261346},"TEMPERATURE":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":3.64,"\\u6807\\u51c6\\u5dee":0.0547722558,"\\u6700\\u5c0f\\u503c":3.6,"\\u4e2d\\u4f4d\\u6570":3.6,"\\u6700\\u5927\\u503c":3.7,"\\u65b9\\u5dee":0.003,"\\u5cf0\\u5ea6":-3.3333333333,"\\u504f\\u5ea6":0.6085806195,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.015047323},"HUMIDITY":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":95.0,"\\u6807\\u51c6\\u5dee":0.0,"\\u6700\\u5c0f\\u503c":95.0,"\\u4e2d\\u4f4d\\u6570":95.0,"\\u6700\\u5927\\u503c":95.0,"\\u65b9\\u5dee":0.0,"\\u5cf0\\u5ea6":0.0,"\\u504f\\u5ea6":0.0,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.0},"PRESSURE":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":1007.6,"\\u6807\\u51c6\\u5dee":0.5477225575,"\\u6700\\u5c0f\\u503c":1007.0,"\\u4e2d\\u4f4d\\u6570":1008.0,"\\u6700\\u5927\\u503c":1008.0,"\\u65b9\\u5dee":0.3,"\\u5cf0\\u5ea6":-3.3333333333,"\\u504f\\u5ea6":-0.6085806195,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.0005435913},"ROUND(A.WS,1)":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":3.76,"\\u6807\\u51c6\\u5dee":0.1949358869,"\\u6700\\u5c0f\\u503c":3.5,"\\u4e2d\\u4f4d\\u6570":3.7,"\\u6700\\u5927\\u503c":4.0,"\\u65b9\\u5dee":0.038,"\\u5cf0\\u5ea6":-0.8171745152,"\\u504f\\u5ea6":-0.080998291,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.0518446508},"ROUND(A.POWER,0)":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":7197.6,"\\u6807\\u51c6\\u5dee":1381.1684546065,"\\u6700\\u5c0f\\u503c":5773.0,"\\u4e2d\\u4f4d\\u6570":6584.0,"\\u6700\\u5927\\u503c":9272.0,"\\u65b9\\u5dee":1907626.3,"\\u5cf0\\u5ea6":-0.0297561698,"\\u504f\\u5ea6":0.9099822326,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.1918929163},"YD15":{"\\u6837\\u672c\\u91cf":5.0,"\\u5e73\\u5747\\u503c":7148.4,"\\u6807\\u51c6\\u5dee":1433.086284911,"\\u6700\\u5c0f\\u503c":5685.0,"\\u4e2d\\u4f4d\\u6570":6569.0,"\\u6700\\u5927\\u503c":8868.0,"\\u65b9\\u5dee":2053736.3,"\\u5cf0\\u5ea6":-2.7858374898,"\\u504f\\u5ea6":0.4371197529,"\\u53d8\\u5f02\\u7cfb\\u6570(CV)":0.2004765101}}'
     """
     path=request.json
     return data_analysis_process.descriptive_analysis(path)


@app.route("/correlation_analysis",methods=['GET'])
def get_correlation_analysis():
     """
     一些基本统计量
     输入:文件的路径
     输出:json格式
        如：{"TurbID":{"TurbID":null,"WINDSPEED":null,"PREPOWER":null,"WINDDIRECTION":null,"TEMPERATURE":null,"HUMIDITY":null,"PRESSURE":null,"ROUND(A.WS,1)":null,"ROUND(A.POWER,0)":null,"YD15":null},"WINDSPEED":{"TurbID":null,"WINDSPEED":1.0,"PREPOWER":-0.6329527942,"WINDDIRECTION":0.512989176,"TEMPERATURE":0.9128709292,"HUMIDITY":null,"PRESSURE":-0.9128709292,"ROUND(A.WS,1)":0.6412364701,"ROUND(A.POWER,0)":0.7321699222,"YD15":0.8085695971},"PREPOWER":{"TurbID":null,"WINDSPEED":-0.6329527942,"PREPOWER":1.0,"WINDDIRECTION":-0.9732857735,"TEMPERATURE":-0.8662900771,"HUMIDITY":null,"PRESSURE":0.8662900771,"ROUND(A.WS,1)":-0.8921857136,"ROUND(A.POWER,0)":-0.785178071,"YD15":-0.917209202},"WINDDIRECTION":{"TurbID":null,"WINDSPEED":0.512989176,"PREPOWER":-0.9732857735,"WINDDIRECTION":1.0,"TEMPERATURE":0.7492686493,"HUMIDITY":null,"PRESSURE":-0.7492686493,"ROUND(A.WS,1)":0.7763157895,"ROUND(A.POWER,0)":0.6334327919,"YD15":0.8109788224},"TEMPERATURE":{"TurbID":null,"WINDSPEED":0.9128709292,"PREPOWER":-0.8662900771,"WINDDIRECTION":0.7492686493,"TEMPERATURE":1.0,"HUMIDITY":null,"PRESSURE":-1.0,"ROUND(A.WS,1)":0.889756521,"ROUND(A.POWER,0)":0.9037707923,"YD15":0.9711648442},"HUMIDITY":{"TurbID":null,"WINDSPEED":null,"PREPOWER":null,"WINDDIRECTION":null,"TEMPERATURE":null,"HUMIDITY":null,"PRESSURE":null,"ROUND(A.WS,1)":null,"ROUND(A.POWER,0)":null,"YD15":null},"PRESSURE":{"TurbID":null,"WINDSPEED":-0.9128709292,"PREPOWER":0.8662900771,"WINDDIRECTION":-0.7492686493,"TEMPERATURE":-1.0,"HUMIDITY":null,"PRESSURE":1.0,"ROUND(A.WS,1)":-0.889756521,"ROUND(A.POWER,0)":-0.9037707923,"YD15":-0.9711648442},"ROUND(A.WS,1)":{"TurbID":null,"WINDSPEED":0.6412364701,"PREPOWER":-0.8921857136,"WINDDIRECTION":0.7763157895,"TEMPERATURE":0.889756521,"HUMIDITY":null,"PRESSURE":-0.889756521,"ROUND(A.WS,1)":1.0,"ROUND(A.POWER,0)":0.9650522872,"YD15":0.9615551689},"ROUND(A.POWER,0)":{"TurbID":null,"WINDSPEED":0.7321699222,"PREPOWER":-0.785178071,"WINDDIRECTION":0.6334327919,"TEMPERATURE":0.9037707923,"HUMIDITY":null,"PRESSURE":-0.9037707923,"ROUND(A.WS,1)":0.9650522872,"ROUND(A.POWER,0)":1.0,"YD15":0.9537430826},"YD15":{"TurbID":null,"WINDSPEED":0.8085695971,"PREPOWER":-0.917209202,"WINDDIRECTION":0.8109788224,"TEMPERATURE":0.9711648442,"HUMIDITY":null,"PRESSURE":-0.9711648442,"ROUND(A.WS,1)":0.9615551689,"ROUND(A.POWER,0)":0.9537430826,"YD15":1.0}}
     """
     path=request.json
     return data_analysis_process.correlation_analysis(path)

@app.route("/data_preprocess",methods=['GET'])
def get_data_preprocess():
     """
     进行数据处理
     输入:文件的路径，before_impute_method,detection,after_impute_method
     输出:处理后的文件路径
     """
     file_path,before_impute_method,detection,after_impute_method=request.json.split("?")
     return data_analysis_process.data_preprocess(file_path, before_impute_method, detection, after_impute_method)

if __name__ == '__main__':
    app.run()
