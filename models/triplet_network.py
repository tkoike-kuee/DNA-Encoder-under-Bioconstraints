from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons.losses as tfa_loss
import numpy as np
import pandas as pd

class TripletNetwork:
    defaults = {
        "batch_size": 128,
        "margin": 0.8,
        "homopolymer": 3.0,
    }
    def __init__(self, encoder, **kwargs):
        for arg, val in self.defaults.items():
            setattr(self, arg, val)

        for arg, val in kwargs.items():
            setattr(self, arg, val)

        self.encoder = encoder

        anchor = layers.Input(shape=(encoder.input_dim,), name="input_anchor", batch_size=self.batch_size)

        anchor_dna = encoder(anchor)

        anchor_onehots = layers.Lambda(lambda x: x*tf.one_hot(tf.math.argmax(x,2),4))(anchor_dna)

        self.model = tf.keras.Model(inputs=anchor, outputs=anchor_onehots)

        self.model.summary()
        self.triplet_loss = tfa_loss.TripletSemiHardLoss(margin=self.margin, distance_metric=self.pairwise_distance)
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(1e-2), loss=self.custom_loss)
    
    def custom_loss(self, y_true, y_pred):
        return self.triplet_loss(y_true, y_pred) + self.homopolymer_loss(y_true, y_pred)

    def hamming_distance(self, x, y):
        dist = tf.reduce_sum(tf.abs(tf.subtract(x, y)), [1,2], keepdims=True)/(2*self.encoder.output_len)
        return tf.squeeze(dist,axis=-1)
    
    def pairwise_distance(self, anchor):
        pairwise_distances = tf.reduce_sum(tf.abs(anchor[:, None, :, :] - anchor[None, :, :, :]),axis=[2,3])/(2*self.encoder.output_len)
        num_data = tf.shape(anchor)[0]
        mask_diagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(tf.ones(num_data))
        pairwise_distances = tf.math.multiply(pairwise_distances, mask_diagonals)
        return pairwise_distances

    def homopolymer_loss(self, y_true, y_pred):
        argmax_indices = tf.argmax(y_pred, axis=-1, output_type=tf.int16)
        penalty_sum = tf.map_fn(lambda seq: self.count_run_length(seq), argmax_indices, dtype=tf.float32)
        return tf.reduce_mean(penalty_sum)

    def count_run_length(self, seq_indices):
        mask = tf.concat([tf.constant([True]), tf.equal(seq_indices[1:], seq_indices[:-1])], axis=0)
        states = (tf.constant(1,dtype=tf.float32), tf.constant(0,dtype=tf.float32))
        
        _, penalty_sum = tf.scan(self.calc_penalty, mask, initializer=states)
        return penalty_sum[-1]/self.encoder.output_len

    def calc_penalty(self, state, x):
        prev_run_len, penalty_sum = state
        new_run_len = tf.where(x, prev_run_len + 1.0, 1.0)
        penalty_t = tf.where(tf.logical_and(new_run_len > self.homopolymer, x == False), tf.cast(new_run_len - self.homopolymer, tf.float32), 0.0)

        return (new_run_len, penalty_sum + penalty_t)