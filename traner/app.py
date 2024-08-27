import os
import webbrowser
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO
#from werkzeug.utils import secure_filename
import threading
from flask_cors import CORS
import urllib.parse
import queue
import time

PORT = 8000

app = Flask(__name__, template_folder='./templates')
CORS(app)
socketio = SocketIO(app)

runData = {"graphs": {}, "bars": {}, "chat": []}

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


@socketio.on("disconnect")
def handle_disconnect():
    connected_clients.remove(request.sid)


def emit(name, data):
    socketio.emit(name, data)


@socketio.on("fill_me_in")
def full_data_drop():
    emit("full_data_drop", runData)


class Bar:

    def __init__(self, title):
        self.title = urllib.parse.quote(title)
        runData["bars"][self.title] = {'value': 0, 'color': "steelblue"}
        emit('new_bar', {'title': self.title})

    def update(self, value: float):
        if type(value) in (int, float):
            runData["bars"][self.title]['value'] = value
            emit('new_bar_value', {'title': self.title, 'value': value})

        else:
            raise TypeError("input parameter `value` must be of type int or float.")

    def color(self, value: str):
        if type(value) is str:
            runData["bars"][self.title]['color'] = value
            emit('set_bar_color', {'title': self.title, 'value': value})


class Line:

    def __init__(self, title):
        self.title = urllib.parse.quote(title)
        runData["graphs"][self.title] = {'values': [], 'color': "steelblue"}
        emit('new_graph', {'title': self.title})

    def update(self, value: float):

        if type(value) in (int, float):
            runData["graphs"][self.title]['values'].append(value)
            emit('new_graph_value', {'title': self.title, 'value': value})

        else:
            raise TypeError("input parameter `value` must be of type int or float.")


    def color(self, value: str):
        if type(value) is str:
            runData["graphs"][self.title]['color'] = value
            emit('set_graph_color', {'title': self.title, 'value': value})



def send_lm_message(string):
    runData["chat"].append({'message': string, 'sender': 0})
    emit('lm_message', {'message': string})


@socketio.on("user_message")
def receive_user_message(data):
    runData["chat"].append({'message': data["message"], 'sender': 1})
    response = lm(data["message"])
    send_lm_message(response)



def lm(string):
    response = string[::-1]
    time.sleep(5)
    return response


def run_flask():
    socketio.run(app, allow_unsafe_werkzeug=True, port=PORT)


def init(**kwargs):

    threading.Thread(target=run_flask).start()
    webbrowser.open_new(f"http://127.0.0.1:{PORT}")
    print("hello")
