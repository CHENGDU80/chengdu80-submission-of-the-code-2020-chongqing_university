from flask import Blueprint, request, render_template
from view.model_arima import getArimaData
from view.eval_model import get_lstm_prediction
from models.kd import eval_model
import json

static_view_b = Blueprint('web_access', __name__)


@static_view_b.route('/')
@static_view_b.route('/LSTM')
def get_LSTM():
    return render_template('index.html')


@static_view_b.route('/ARIMA')
def get_arima():
    return render_template('index2.html')


@static_view_b.route('/KnowledgeDistillation')
def get_KnowlegeDistillation():
    return render_template('index3.html')


@static_view_b.route('/TimeLine')
def get_time_line():
    return render_template('timeline.html')


@static_view_b.route('/ContactList')
def get_contact_list():
    return render_template('contact-list.html')


@static_view_b.route('/static_k_data', methods=["POST", "GET"])
def get_data():
    name = request.form["company_name"]
    name = name.upper()
    origin_json_path = "data/company_json_data/{}.json".format(name)
    with open(origin_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data)


@static_view_b.route('/predict_company_data', methods=["POST", "GET"])
def post_alone_company():
    name = request.form['company_name']
    name = name.upper()
    all_data = {}
    predict_data = getArimaData(name)
    all_data['predict'] = predict_data.tolist()
    orgin_json_path = "data/company_json_data/{}.json".format(name)
    with open(orgin_json_path, encoding="utf-8") as f:
        origin_data = json.load(f)
        all_data['orgin'] = origin_data
    return json.dumps(all_data)


@static_view_b.route('/predict_company_data_lstm', methods=["POST", "GET"])
def predict_company_data_lstm():
    print(request.form)
    name = request.form["company_name"]
    name = name.upper()
    all_data = {}
    _, _, _, predict_data = get_lstm_prediction(name)
    all_data['predict'] = predict_data
    orgin_json_path = "data/company_json_data/{}.json".format(name)
    with open(orgin_json_path, encoding="utf-8") as f:
        origin_data = json.load(f)
        all_data['orgin'] = origin_data
    return json.dumps(all_data)


@static_view_b.route('/show_company_relation', methods=["POST", "GET"])
def show_company_relation():
    name = request.form["company_name"]
    name = name.upper()
    with open("data/relation_json_data/company_correlation.json") as f:
        data = json.load(f)
        one_data = data[name]
        dict = {}
        for i, item in enumerate(one_data):
            if(i):
                dict[item[0]] = item[1]
                if(i > 20):
                    break
        return json.dumps(dict)


@static_view_b.route('/news', methods=["POST", "GET"])
def get_news():
    origin_json_path = "data/news/2012-12-31.json"
    with open(origin_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data)


@static_view_b.route('/show_kd', methods=["POST", "GET"])
def predict_company_data_kd():
    print(request.form)
    name = request.form["company_name"]
    name = name.upper()
    all_data = {}
    _, kernel, _ = eval_model.get_ks__prediction(name)

    all_data['features'] = [
        "BIDLO", "ASKHI", "OPENPRC", "VOL", "SHROUT", "POLARITY",
        "SUBJECTIVITY", "MA10", "MA20", "MA30", "DIFF", "DEA", "MACD", "RSI6",
        "RSI12", "RSI24", "MFI"
    ]
    all_data['weight'] = kernel

    return json.dumps(all_data)
