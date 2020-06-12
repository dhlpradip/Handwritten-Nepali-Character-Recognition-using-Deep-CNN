from __future__ import division, print_function
import numpy as np # linear algebras 
#import cv2 as cv
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import misc
import imageio
import matplotlib.pyplot as plt
import os
import random
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
import numpy as np
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from PIL import Image
import cv2
import numpy as np # linear algebras 
import matplotlib.pyplot as plt
import seaborn as sns
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
#Play sound 
from playsound import playsound

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'E:/Final Project/nadam.hdf5'

# Load your trained model
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('nadam.hdf5',compile=False)
model._make_predict_function()          
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')
'''
def threshold(imageArray):

    balanceAr=[]
    newAr = imageArray
    from statistics import mean
    for eachRow in imageArray:
        for eachPix in eachRow:
                avgNum = mean(eachPix[:3])
                balanceAr.append(avgNum)

    balance = mean(balanceAr)
    for eachRow in newAr:
        for eachPix in eachRow:
            if (mean(eachPix[:3])<balance) :
                eachPix[0] = 255
                eachPix[1] = 255                
                eachPix[2] = 255
            else:
                eachPix[0] = 0
                eachPix[1] = 0
                eachPix[2] = 0

    return imageArray

'''
def model_predict(img_path,model):
    # Preprocessing the image
    size=500
    img = cv2.imread(img_path)
    img = cv2.resize(img,(size,size), interpolation = cv2.INTER_AREA)
    ret, thresh = cv2.threshold(img, 85, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(thresh)
    plt.show()
    cv2.imwrite('iar.png',thresh)
    img=cv2.imread('iar.png',cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(32,32), interpolation = cv2.INTER_AREA)
    img=np.array(img).reshape(-1,32,32,1)
    img_class = model.predict_classes(img)
    return img_class


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        	basepath,'uploads', secure_filename(f.filename))
        f.save(file_path)
        

        # Make prediction
        preds = model_predict(file_path,model)
        if preds== 0:
            preds='SHUNYA'
            playsound('./voice/shunya.mp3')
        elif preds== 1:
        	preds='EK'
        	playsound('./voice/ek.wav')
        elif preds== 2:
        	preds='DUI' 
        	playsound('./voice/dui.wav')
        elif preds== 3:
        	preds='TIN'
        	playsound('./voice/tin.wav')
        elif preds== 4:
        	preds='CHAR'
        	playsound('./voice/char.wav')
        elif preds== 5:
        	preds='PAANCH'
        	playsound('./voice/panch.wav')
        elif preds== 6:
        	preds='CHHA ANKA'
        	playsound('./voice/chha.wav')
        elif preds== 7:
        	preds='SAAT'
        	playsound('./voice/saat.wav')
        elif preds== 8:
        	preds='AATH'
        	playsound('./voice/aath.wav')
        elif preds== 9:
        	preds='NAU'
        	playsound('./voice/nau.wav')
        elif preds== 10:
        	preds='KA'
        	playsound('./voice/ka.wav')
        elif preds== 11:
        	preds='KHA'
        	playsound('./voice/kha.wav')
        elif preds== 12:
        	preds='GA'
        	playsound('./voice/ga.wav')
        elif preds== 13:
        	preds='GHA'
        	playsound('./voice/gha.wav')
        elif preds== 14:
        	preds='NCHA'
        	playsound('./voice/nca.wav')
        elif preds== 15:
        	preds='CHA'
        	playsound('./voice/cha.wav')
        elif preds== 16:
        	preds='CHHAA'
        	playsound('./voice/chha.wav')
        elif preds== 17:
        	preds='JA'
        	playsound('./voice/ja.wav')
        elif preds== 18:
        	preds='JHA'
        	playsound('./voice/jha.wav')
        elif preds== 19:
        	preds='YAN'
        	playsound('./voice/ya.wav')
        elif preds== 20:
        	preds='TA'
        	playsound('./voice/ta.wav')
        elif preds== 21:
        	preds='THA'
        	playsound('./voice/tha.wav')
        elif preds== 22:
        	preds='DA'
        	playsound('./voice/da.wav')
        elif preds== 23:
        	preds='DHA'
        	playsound('./voice/dha.wav')
        elif preds== 24:
        	preds='NDA'
        	playsound('./voice/nda.wav')
        elif preds== 25:
        	preds='TARAL TA'
        	playsound('./voice/taral.wav')
        elif preds== 26:
        	preds='THARMAS THA'
        	playsound('./voice/tharmas.wav')
        elif preds== 27:
        	preds='DAILO DA'
        	playsound('./voice/dailo.wav')
        elif preds== 28:
        	preds='DHANU DHA'
        	playsound('./voice/dhanu.wav')
        elif preds== 29:
        	preds='NA'
        	playsound('./voice/na.wav')
        elif preds== 30:
        	preds='PA'
        	playsound('./voice/pa.wav')
        elif preds== 31:
        	preds='FA'
        	playsound('./voice/fa.wav')
        elif preds== 32:
        	preds='BA'
        	playsound('./voice/ba.wav')
        elif preds== 33:
        	preds='BHA'
        	playsound('./voice/bha.wav')
        elif preds== 34:
        	preds='MA'
        	playsound('./voice/ma.wav')
        elif preds== 35:
        	preds='BUDHO YA'
        	playsound('./voice/yogi.wav')
        elif preds== 36:
        	preds='RA'
        	playsound('./voice/ra.wav')
        elif preds== 37:
        	preds='LA'
        	playsound('./voice/la.wav')
        elif preds== 38:
        	preds='WA'
        	playsound('./voice/wa.wav')
        elif preds== 39:
        	preds='DANTYA SHA'
        	playsound('./voice/sa.wav')
        elif preds== 40:
        	preds='MURDHANYA SHAW'
        	playsound('./voice/sa.wav')
        elif preds== 41:
        	preds='TALABYA SA'
        	playsound('./voice/sa.wav')
        elif preds== 42:
        	preds='HA'
        	playsound('./voice/ha.wav')
        elif preds== 43:
        	preds='KSHYA'
        	playsound('./voice/kshya.wav')
        elif preds== 44:
        	preds='TRA'
        	playsound('./voice/tra.wav')
        elif preds== 45:
        	preds='GYA'
        	playsound('./voice/gya.wav')
        else:
            preds='INVALID ENTRY'
            playsound('./voice/amanya.wav')
        return preds

    return None


if __name__ == '__main__':
    app.run(debug=True)

