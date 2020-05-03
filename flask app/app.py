import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import scipy
import pickle
import cv2
from PIL import Image
import os

#Keras
from keras.layers import Input, Activation, Dense, Conv2D, Reshape, concatenate, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras import optimizers

#flask 
from flask import Flask, render_template, url_for,redirect,request
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

#load model
CATEGORIES    = ['Mononuclear', 'Polynuclear']
importModel = tf.keras.models.load_model(r'C:\Users\arifm\OneDrive\Documents\PRJ400\blood cells\model_cell.h5')





@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        #read image file string data
        filestr = request.files['file']#.read()

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(filestr.filename))
        filestr.save(file_path)

        img = cv2.imread(file_path)

        # convert string of image data to uint8
        #nparr = np.fromstring(filestr, np.uint8)
        # decode image
        #img2 = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

        #resize image
        img_file = Image.fromarray(img).resize(size=(128, 128))
        #convert to array
        img_file = np.array(img_file)

        #normalize the image
        img_file = img_file.astype('float32') / 255

        #convert to 4 demensional array
        img_file = np.array(img_file).reshape(-1, 128, 128, 3)

        #make predictions
        predictions = importModel.predict(img_file)


        #value = process_image(img2)

        #convert string data to numpy array
        #npimg = np.fromstring(filestr, np.uint8)
        # convert numpy array to image
        #img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        #img_file = cv2.resize(img,(128, 128))
        #X_pred = np.array(img_file)
        #X_pred = np.array(X_pred).reshape(-1, 128, 128, 3)
        #X_pred = X_pred.astype('float32') / 255
        # Make prediction
        #print(type(X_pred),X_pred.shape)
        #predictions = importModel.predict(X_pred)

        # Process your result for human
        #predicted_val = [int(round(p[0])) for p in predictions]
        #results = predicted_val[0]   # ImageNet Decode
        #print(results)
        #result = CATEGORIES[results]               # Convert to string
        #return value
        #pred value
        val = predictions[0]
        num = np.rint(val)
        return num
    return None

if __name__ == "__main__":
    app.run(debug=True)