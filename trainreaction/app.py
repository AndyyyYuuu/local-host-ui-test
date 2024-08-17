import os
import numpy as np
import webbrowser
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO
#from werkzeug.utils import secure_filename
import threading
from flask_cors import CORS
import urllib.parse
import queue

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


request_queue = queue.Queue()
connected_clients = set()



@socketio.on("connect")
def handle_connect():
    connected_clients.add(request.sid)
    while not request_queue.empty():
        data = request_queue.get()
        socketio.emit(*data)


@socketio.on("disconnect")
def handle_disconnect():
    connected_clients.remove(request.sid)


def emit(name, data):
    if connected_clients:
        socketio.emit(name, data)
    else:
        request_queue.put((name, data))



class Bar:

    def __init__(self, title):
        self.title = urllib.parse.quote(title)
        emit('new_bar', {'title': self.title})

    def update(self, value: float):
        if type(value) in (int, float):
            statistics["bar"][self.title] = value

            emit('new_bar_value', {'title': self.title, 'value': value})

        else:
            raise TypeError("input parameter `value` must be of type int or float.")

    def color(self, value: str):
        if type(value) is str:
            emit('set_bar_color', {'title': self.title, 'value': value})


class Line:

    def __init__(self, title):
        self.title = urllib.parse.quote(title)
        emit('new_graph', {'title': self.title})

    def update(self, value: float):

        if type(value) in (int, float):
            if self.title in statistics["graph"].keys():
                statistics["graph"][self.title].append(value)
            else:
                statistics["graph"][self.title] = [value]

            emit('new_graph_value', {'title': self.title, 'value': value})

        else:
            raise TypeError("input parameter `value` must be of type int or float.")


    def color(self, value: str):
        if type(value) is str:
            emit('set_graph_color', {'title': self.title, 'value': value})


def send_lm_message(string):
    emit('lm_message', {'message': string});


def run_flask():
    socketio.run(app, allow_unsafe_werkzeug=True, port=PORT)


def init(**kwargs):

    threading.Thread(target=run_flask).start()
    webbrowser.open_new(f"http://127.0.0.1:{PORT}")
    print("hello")
