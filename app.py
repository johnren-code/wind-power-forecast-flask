from flask import Flask,request
import pred
app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route("/pred",methods=['GET'])
def pred():
    info=request.json
    tid,start,len1=info.split('?')
    return pred.pred(tid,start,len1)

if __name__ == '__main__':
    app.run()
