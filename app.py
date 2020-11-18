import os

from flask import Flask, render_template, request, redirect, flash
from keras_preprocessing.image import load_img, img_to_array
from tensorflow import keras
from tensorflow.keras import layers
from werkzeug.utils import secure_filename
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

labels = {
    0: "NORMAL",
    1: "PNEUMONIA BACTERIA",
    2: "PNEUMONIA VIRUS"
}


def load_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 70, 1)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.load_weights("models/model.80-0.39.h5")
    return model


def getPrediction(filename):
    model = load_model()
    image = load_img('uploads/' + filename, target_size=(50, 70), color_mode="grayscale")
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    pred = model.predict(image)[0]
    label = labels[np.argmax(pred)]
    prob = np.max(pred)
    return label, prob


@app.route('/')
def index():
    model = load_model()
    return render_template("index.html", model=model)


# Source: https://gist.github.com/mrron313/41f5691b4066876103bcfa77e6ccc065#file-index-py
@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label, prob = getPrediction(filename)
            flash(f"{label}")
            flash(f"{prob}")
            return redirect('/')


if __name__ == "__main__":
    app.run()
