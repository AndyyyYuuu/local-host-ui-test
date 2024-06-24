import os
import numpy as np
import shap
import torch
import matplotlib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO, emit
#from werkzeug.utils import secure_filename
import threading
from multiprocessing import Process, freeze_support
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

statistics = {"graph":{}, "bar":{}, "number":{}}

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'DoubleDoubleToilAndTrouble'

current_model = None


def get_model_stats(model: torch.nn.Module):
    return jsonify(text="Stats")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route("/")
def index():
    return render_template("index.html")


def update_bar(title, value: float):
    if type(value) in (int, float):
        statistics["bar"][title] = value

        socketio.emit('new_bar_value', {'title': title, 'value': value})

    else:
        raise TypeError("Input parameter `value` must be of type int or float.")


def update_graph(title, value: float):
    if type(value) in (int, float):
        if title in statistics["graph"].keys():
            statistics["graph"][title].append(value)
        else:
            statistics["graph"][title] = [value]

        socketio.emit('new_graph_value', {'title': title, 'value': value})

    else:
        raise TypeError("Input parameter `value` must be of type int or float.")


def run_flask():
    socketio.run(app, allow_unsafe_werkzeug=True, port=8000)


def init(**kwargs):

    threading.Thread(target=run_flask).start()

    print("hello")
