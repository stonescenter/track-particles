__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import os
import math
import datetime as dt

import tensorflow as tf

import keras.backend as K
from keras.backend.tensorflow_backend import set_session

from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model   

from core.utils.utils import Timer
import numpy as np

import shortuuid
import uuid

def get_unique_name(name):

    shortuuid.set_alphabet("0123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    shortuuid.ShortUUID().random(length=16)
    uid = uuid.uuid5(uuid.NAMESPACE_DNS, name)
    enc = shortuuid.encode(uid)
    return enc

def get_decryp_name(key):
    dec = shortuuid.decode(key)
    return dec

class BaseModel():
    def __init__(self, configs):
        self.model = Sequential()
        self.name = configs['model']['name']
        self.normalise = configs['data']['normalise']
        self.cylindrical = configs['data']['cylindrical']

        self.orig_ds_name = configs['data']['filename']
        self.encryp_ds_name = get_unique_name(self.orig_ds_name)
        self.decryp_ds_name = get_decryp_name(self.encryp_ds_name)


        if self.cylindrical:
            coord = 'cylin'
        else:
            coord = 'xyz'

        self.save_fnameh5 = os.path.join(configs['paths']['save_dir'], 
            'model-%s-%s-coord-%s-normalise-%s.h5' % (self.name, self.encryp_ds_name, coord,
                str(self.normalise).lower() ))
        print(self.save_fnameh5)

        self.save_fname = os.path.join(configs['paths']['save_dir'], 'architecture-%s.png' % self.name)

        self.save = configs['training']['save_model']
        
        if configs['training']['use_gpu'] == True:
            #config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 0} ) 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config) 
            set_session(sess)
            tf.device('/gpu:0')
        else:
            config=tf.ConfigProto(log_device_placement=True)
            sess = tf.Session(config=config)
            set_session(sess)
        
        #set_random_seed(42)
        tf.compat.v1.set_random_seed(0)

    def load_model(self):
        if self.exist_model(self.save_fnameh5):
            print('[Model] Loading model from file %s' % self.save_fnameh5)
            self.model = load_model(self.save_fnameh5)
            return True
        else:
            print('[Model] Can not load the model from file %s' % self.save_fnameh5)
        return False
    
    def exist_model(self, filepath):
        if os.path.exists(filepath): 
            return True
        return False

    def save_architecture(self, filepath):

        plot_model(self.model, to_file=filepath, show_shapes=True)
        print('[Model] Model Architecture saved at %s' % filepath)

    def save_model(self, filepath):
        self.model.save(filepath)
        print('[Model] Model for inference saved at %s' % filepath)

    def train(self, x, y, epochs, batch_size):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        #print('[Model] Shape of data train: ', x.shape) 
        #save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=self.save_fnameh5, monitor='val_loss', save_best_only=True)
        ]
        history = self.model.fit(
            x,
            y,
            validation_split=0.33,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        if self.save == True:
            self.save_model(self.save_fnameh5)             
       
        print('[Model] Training Completed. Model h5 saved as %s' % self.save_fnameh5)
        print('[Model] Model train with structure:', self.model.inputs)
        timer.stop()

        return history
    
    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
         
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )
         
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def evaluate(self, x, y, batch_size=10):
        results = self.model.evaluate(x, y, batch_size=batch_size, verbose=2)
        print('[Model] Test loss %s accuracy %s :' %(results[0], results[1]))
        
    def predict_one_hit(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time

        print('[Model] Predicting Hit-by-Hit...')
        predicted = self.model.predict(data)
        print('[Model] Predicted shape predicted%s size %s' % (predicted.shape, predicted.size))

        #predicted = np.reshape(predicted, (predicted.size, 1))
        return predicted
 
    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs
 
    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted


