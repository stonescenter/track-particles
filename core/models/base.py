

import os
import math
import datetime as dt


from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model   

from core.utils.utils import Timer
import numpy as np

class BaseModel():
    def __init__(self):
        self.model = Sequential()
        self.name = ''
 
    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def save_architecture(self, filepath):
        plot_model(self.model, to_file=filepath, show_shapes=True)
        print('[Model] Model Architecture saved at %s' % filepath)

    def save_model(self, filepath):
        self.model.save(filepath)
        print('[Model] Model for inference saved at %s' % filepath)

    def train(self, x, y, epochs, batch_size, save_fname):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
         
        #save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        history = self.model.fit(
            x,
            y,
            validation_split=0.33,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        self.save_model(save_fname)
        
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        print('[Model] Model tr with structure:', self.model.inputs)
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

        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        print('[Model] Shape predict result %s size %s' % (predicted.shape, predicted.size))

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


