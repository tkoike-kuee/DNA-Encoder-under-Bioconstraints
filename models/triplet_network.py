from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons.losses as tfa_loss
import numpy as np
import pandas as pd
import math

class TripletNetwork:
    defaults = {
        "batch_size": 128,
        "margin": 0.8,
        "homopolymer": 4,
        "hp_scale": 0.01
    }
    def __init__(self, encoder, **kwargs):
        for arg, val in self.defaults.items():
            setattr(self, arg, val)

        for arg, val in kwargs.items():
            setattr(self, arg, val)
        
        print(f"TripletNetwork: {self.__dict__}")

        self.encoder = encoder

        anchor = layers.Input(shape=(encoder.input_dim,), name="input_anchor", batch_size=self.batch_size)

        anchor_dna = encoder(anchor)

        self.model = tf.keras.Model(inputs=anchor, outputs=anchor_dna)

        self.model.summary()

        self.triplet_loss = tfa_loss.TripletSemiHardLoss(margin=self.margin, distance_metric=self.pairwise_distance)
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(1e-2), loss=self.custom_loss, metrics=[self.homopolymer_metrics])
    
    def custom_loss(self, y_true, y_pred):
        t_loss = self.triplet_loss(y_true, y_pred)
        h_loss = self.weighted_homopolymer_loss(y_true, y_pred)
    
        return t_loss + h_loss*self.hp_scale

    def hamming_distance(self, x, y):
        dist = tf.reduce_sum(tf.abs(tf.subtract(x, y)), [1,2], keepdims=True)/(2*self.encoder.output_len)
        return tf.squeeze(dist,axis=-1)
    
    def homopolymer_loss(self, y_true, y_pred):
        l2_norm = tf.math.l2_normalize(y_pred, axis=-1)
        similarity = tf.reduce_sum(l2_norm[:,1:,:] * l2_norm[:,:-1,:], axis=-1)
        return tf.reduce_mean(similarity)

    def weighted_homopolymer_loss(self, y_true, y_pred):
        l2_norm = tf.math.l2_normalize(y_pred, axis=-1)
        similarity = tf.reduce_sum(l2_norm[:,1:,:] * l2_norm[:,:-1,:], axis=-1)
        scale_value = math.sqrt(2)*similarity # angle to distance
        split_frame = tf.signal.frame(scale_value, frame_length=self.homopolymer, frame_step=1, axis=-1, pad_end=False)
        log_prod = tf.reduce_mean(tf.math.log(split_frame + tf.keras.backend.epsilon()), axis=-1)
        return tf.reduce_mean(tf.exp(log_prod))
    
    def homopolymer_metrics(self, y_true, y_pred):
        argmax_indices = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        equal_list = [tf.equal(argmax_indices, tf.roll(argmax_indices,shift=i,axis=-1)) for i in range(1, 2)]
        mask = tf.reduce_all(tf.stack(equal_list, axis=-1), axis=-1)
        batch_size = tf.shape(y_true)[0]
        mask = tf.concat([tf.zeros((batch_size, 1), dtype=tf.bool), mask[:,1:]], axis=-1)
        masked = tf.ragged.boolean_mask(argmax_indices, mask)
        onehots =tf.one_hot(masked, 4)
        penalty = tf.reduce_sum(onehots, axis=[1,2])
        return tf.reduce_mean(penalty)
    
    def pairwise_distance(self, features):
        anchor = features*tf.one_hot(tf.math.argmax(features,2),4)
        pairwise_distances = tf.reduce_sum(tf.abs(anchor[:, None, :, :] - anchor[None, :, :, :]),axis=[2,3])/(2*self.encoder.output_len)
        num_data = tf.shape(anchor)[0]
        mask_diagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(tf.ones(num_data))
        pairwise_distances = tf.math.multiply(pairwise_distances, mask_diagonals)
        return pairwise_distances

    # will implement unsorted_segment_sum for gc content loss
