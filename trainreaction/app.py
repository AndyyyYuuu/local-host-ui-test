import os
import numpy as np
import webbrowser
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO, emit
#from werkzeug.utils import secure_filename
import threading
from flask_cors import CORS
import urllib.parse

PORT = 8000

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


class bar:

    @staticmethod
    def update(title: str, value: float):
        title = urllib.parse.quote(title)
        if type(value) in (int, float):
            statistics["bar"][title] = value

            socketio.emit('new_bar_value', {'title': title, 'value': value})

        else:
            raise TypeError("input parameter `value` must be of type int or float.")


class line:

    @staticmethod
    def update(title: str, value: float):
        title = urllib.parse.quote(title)
        if type(value) in (int, float):
            if title in statistics["graph"].keys():
                statistics["graph"][title].append(value)
            else:
                statistics["graph"][title] = [value]

            socketio.emit('new_graph_value', {'title': title, 'value': value})

        else:
            raise TypeError("input parameter `value` must be of type int or float.")


def run_flask():
    socketio.run(app, allow_unsafe_werkzeug=True, port=PORT)


def init(**kwargs):

    threading.Thread(target=run_flask).start()
    webbrowser.open_new(f"http://127.0.0.1:{PORT}")
    print("hello")
