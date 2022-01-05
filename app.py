# _*_ coding: UTF-8 _*_
# Author LBK
from flask import Flask, request
import json
from model import Classifier

# 初始化模型, 避免在函数内部初始化,耗时过长
bc = Classifier()
bc.load()

# 初始化flask
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def gen_ans():
    result = {}

    text = request.form["text"]
    label = bc.predict(text)
    result = {
        "label": label
    }
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)