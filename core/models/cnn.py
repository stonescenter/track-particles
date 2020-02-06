from keras.layers import Input, Dense, Activation, Dropout, LSTM
from keras.layers import concatenate, CuDNNLSTM
from keras.models import Model

from .base import BaseModel
from .timer import Timer

class ModelCNN(BaseModel):
    """A class for an building and inferencing an lstm model"""
 
    def __init__(self):
        super().__init__()
 
    def build_model(self, configs):
        timer = Timer()
        timer.start()
        
        print('[Model] Creating model..')    

        # this model is not sequencial
        self.model = None

        for layer in configs['model']['layers']:

            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_features = layer['input_features'] if 'input_features' in layer else None

            #create Neural Network
            if layer['type'] == 'lstm-paralel':

                first_input = Input(shape=(input_timesteps, input_features))        
                first_output = CNN(neurons, return_sequences=return_seq, return_state=False)(first_input)

                second_input = Input(shape=(input_timesteps, 1)) # without number of features, just with input_timesteps
                second_output = CNN(neurons, return_sequences=return_seq, return_state=False)(second_input)

                output = concatenate([first_output, second_output], axis=-1)

            if layer['type'] == 'dense':
                output = Dense(neurons, activation = activation)(output)
            if layer['type'] == 'dropout':
                output = Dropout(0.5)(output)

        self.model = Model(inputs=[first_input, second_input], outputs=output)
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=['accuracy'])
        
        print('[Model] Model Compiled with structure:', self.model.inputs)      
        timer.stop()
