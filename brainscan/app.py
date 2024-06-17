import os
import numpy as np
import shap
import torch
import matplotlib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        print("No selected file")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File successfully uploaded')
        print("Success")
        return redirect(url_for('index'))
    else:
        flash('Allowed file types are png, jpg, jpeg, gif')
        return redirect(request.url)


def init(model: torch.nn.Module, input_shape: torch.Size):
    print(list(model.children())[0])
    shap_shape = torch.Size([100]+list(input_shape))
    x = torch.tensor(torch.randn(shap_shape)).float()
    explainer = shap.DeepExplainer(model, x)
    shap.summary_plot(explainer.shap_values(x), features=x)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)