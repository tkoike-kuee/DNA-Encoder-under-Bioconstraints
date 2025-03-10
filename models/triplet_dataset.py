import tensorflow as tf
import numpy as np

class TripletDataset:
    defaults = {
        "batch_size": 128,
        "margin": 0.8,
        "num_classes": 10
    }

    def __init__(self, encoder, features, labels, **kwargs):
        for arg, val in self.defaults.items():
            setattr(self, arg, val)

        for arg, val in kwargs.items():
            setattr(self, arg, val)

        self.features = tf.convert_to_tensor(features)
        self.labels = tf.convert_to_tensor(labels)
        self.data_size = features.shape[0]
        # データセットの作成
        self.dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels)).shuffle(buffer_size=self.data_size).batch(self.batch_size)

        