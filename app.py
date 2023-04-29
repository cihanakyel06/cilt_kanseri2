from __future__ import print_function


import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
from skimage.measure import label, regionprops
import skimage.io as io
import skimage.transform as trans
from keras.initializers import Constant
from keras.layers import Activation, Dense
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error
from skimage.morphology import disk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.applications.vgg19 import VGG19, preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16
from cv2 import imwrite, resize
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose, Average,Dropout
from tensorflow.keras.layers import Dense, Dropout, add

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau,CSVLogger

import cv2
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras


r=192
c=192

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
import pickle
def conv_block(inputs, num_filters):
    skip=inputs
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x) 
    x = Dropout(0.5)(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)  
    x = Dropout(0.5)(x)
    x = Concatenate()([x, skip])
    
    

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x
def encoder_block(inputs, num_filters):
    x=conv_block(inputs,num_filters)

    p=MaxPool2D((2,2))(x)
    return x,p

def build_effienet_unet():
    """ Input """
    """ Input """
    """ Input """
    inputs = Input(shape=(192,192,3))

    """ Encoder """
    s1,p1 = encoder_block(inputs,32)                      ## 256
    s2,p2 = encoder_block(p1,64)      ## 128
    s3,p3 = encoder_block(p2,128)     ## 64
    s4,p4 = encoder_block(p3,256)     ## 32

    b1 = conv_block(p4,512)    ## 16

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model
    
def sinif():  
    # Defining base model using Xception module from Keras
    training_shape = (192,192,3)
    base_model = ResNet50(include_top=False,weights='imagenet',input_shape = training_shape)
    for layer in base_model.layers:
        layer.trainable = True        
#Adding layers at end
    n_classes = 2
    model = base_model.output
    model = Flatten()(model)
    model = Dense(128)(model)
    model = BatchNormalization()(model)
    model = Dropout(0.3)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.4)(model)
    output = Dense(n_classes, activation='sigmoid')(model)
    model = Model(inputs=base_model.input, outputs=output)
    return model
    
def adjustData(img, mask):

        
    img = preprocess_input(img)
    img = img/img.max() 
    mask = mask/mask.max()
    return (img, mask)
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# def bce_dice_loss(y_true, y_pred):
#     return binary_crossentropy(y_true, K.sigmoid(y_pred)) + dice_loss(y_true, K.sigmoid(y_pred))

def mean_squared_error_dice_loss(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred) + dice_coef_loss(y_true, y_pred))

model= build_effienet_unet()
modelsinif=sinif()

from flask import Flask,  url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__, static_url_path='')

 
app.config['UPLOAD_FOLDER'] = 'uploads'
# Model saved with Keras model.save()



#Load your trained model
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
import keras.losses
keras.losses.custom_loss=mean_squared_error_dice_loss
import keras.losses
keras.losses.custom_loss=dice_coef_loss
import keras.metrics
keras.metrics.custom_objects=dice_coef

model = load_model('D:/PROJELER/TEZ/kilYeniSaved2.hdf5', custom_objects={'mean_squared_error_dice_loss': mean_squared_error_dice_loss,'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef})
modelsinif=load_model("D:/DOKTORAS/TEZ/BOLUTLU_SINIF.hdf5", custom_objects={'mean_squared_error_dice_loss': mean_squared_error_dice_loss,'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef}) 

@app.route('/uploads/<filename>')

def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


def model_predict(img_path, model):
  
   img = cv2.imread(img_path,1) 
   X=img
   Y=[]
   X=cv2.resize(X,(192,192))
   Y.append(X)
   Y=np.array(Y)
   Y=Y.astype("float32")/255.0

   target_size = (192,192)
   kernel2 = np.ones((2,2),np.uint8)

   img=cv2.resize(img,(192,192))
   img2=img
   img = trans.resize(img,target_size)   
   img=np.expand_dims(img, axis=0)
   
   global sess
   global graph
   with graph.as_default():
    set_session(sess)
    Pr=model.predict(img,verbose=0)
    predictions=modelsinif.predict(Y)
    sonuc=predictions.argmax(axis=1)
   Pr=Pr.reshape(r,c)
   one=255*Pr
   one[one>1]=255
   one=cv2.resize(one,(192,192))
   cv2.imwrite('D:/KDS/Arayuz/temp/sil.jpeg', one)   
   mask = cv2.imread('D:/KDS/Arayuz/temp/sil.jpeg', 0)
   output1=cv2.inpaint(img2,mask,3,cv2.INPAINT_TELEA)
   # os.remove('D:/KDS/Arayuz/temp/'+path[0]+'_sil.jpeg')    
   cv2.imwrite('D:/KDS/Rapor/son5.jpeg' ,output1)
   img=output1
   


   return img,sonuc


@app.route('/predict', methods=['GET', 'POST'])
def predict():
        # Get the file from post request
        
   f = request.files['file']

   # Save the file to ./uploads
   basepath = os.path.dirname(__file__)
   file_path = os.path.join(
       basepath, 'uploads', secure_filename(f.filename))
   print(file_path)
   f.save(file_path)
   file_name=os.path.basename(file_path)
      # Make prediction


   img,sonuc = model_predict(file_path, model)
   cv2.imwrite(file_path+"2.jpeg",img)
   file_name2=os.path.basename(file_path+"2.jpeg")
      
        
   return render_template('predict.html',file_name=file_name,file_name2=file_name2,sonuc=sonuc)


if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
