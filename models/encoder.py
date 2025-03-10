import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from .seqtools import onehots_to_seqs 

class DNAEncoder:

    defaults = {
        "input_dim": 4096,
        "output_len": 80,
        "lstm_units": 32
    }

    def __init__(self, model_path = None, **kwargs):
        for arg, val in self.defaults.items():
            setattr(self, arg, val)

        for arg, val in kwargs.items():
            setattr(self, arg, val)

        if model_path is None:
            self.model = tf.keras.Sequential([
                layers.Dense(int(self.input_dim/2), activation = 'relu', input_shape=[self.input_dim]),
                layers.Dense(self.output_len * 4, activation='relu',name='flat-seq'),
                layers.Reshape([self.output_len, 4],name = "temp_output"),
                layers.LSTM(self.lstm_units, return_sequences=True, input_shape=[self.output_len, 4]),
                layers.Dense(4, activation='softmax')
            ], name='DNA_Encoder')
        else:
            self.model = tf.keras.models.load_model(model_path)
        
        self.model.summary()

    def __call__(self, X):
        return self.model(X)
        
    def trainable(self, flag):
        for layer in self.model.layers:
            layer.trainable = flag

    def encode_feature_seqs(self, X):
        prediction = self.model.predict(X)
        return onehots_to_seqs(prediction)
    
    def save(self, model_path):
        self.model.save(model_path)
    
