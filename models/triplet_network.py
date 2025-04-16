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
    }
    def __init__(self, encoder, hp_scale=1.0, **kwargs):
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
        self.optimizer = tf.keras.optimizers.Adagrad(1e-2)
        self.model.compile(optimizer=self.optimizer, loss=self.custom_loss, metrics=[self.triplet_metrics, self.homopolymer_metrics, self.gc_content_metrics])
    
    def custom_loss(self, y_true, y_pred):
        t_loss = self.triplet_loss(y_true, y_pred)
        h_loss = self.homopolymer_loss(y_true, y_pred)
        gc_loss = self.gc_balance_loss(y_true, y_pred)
        p_penalty = self.prob_penalty(y_true, y_pred)
    
        return t_loss + h_loss + gc_loss + self.encoder.entropy_reg_strength*p_penalty

    def homopolymer_loss(self, y_true, y_pred):
        l2_norm = tf.math.l2_normalize(y_pred, axis=-1)
        cos_similarity = math.sqrt(2) * tf.reduce_sum(l2_norm[:,1:,:] * l2_norm[:,:-1,:], axis=-1)
        split_frame = tf.signal.frame(cos_similarity, frame_length=self.homopolymer, frame_step=1, axis=-1, pad_end=False)
        frame_min = tf.reduce_min(split_frame, axis=-1)
        frame_sum = tf.reduce_sum(split_frame, axis=-1)
        masked_sum = tf.where(frame_min >= 1.0, frame_sum, 0.0)
        penalties = tf.reduce_mean(masked_sum, axis=-1)
        mask = penalties > 0.0
        filtered_penalties = tf.boolean_mask(penalties, mask)

        return tf.cond(tf.size(filtered_penalties) > 0, lambda: tf.reduce_mean(filtered_penalties), lambda: tf.constant(0.0))

    def prob_penalty(self, y_true, y_pred):
        max_prob = tf.reduce_max(y_pred, axis=-1)
        min_prob = tf.reduce_min(y_pred, axis=-1)
        prob_diff = max_prob - min_prob
        log_mean = tf.reduce_mean(-tf.math.log(prob_diff + 1e-8), axis=-1)
        return tf.reduce_mean(log_mean)
    
    def gc_balance_loss(self, y_true, y_pred):
        gc_prob = y_pred[:, :, 0] + y_pred[:, :, 1] 
        mean_gc = tf.reduce_sum(gc_prob, axis=-1)
        penalties =  tf.square(tf.abs(0.5 - mean_gc/self.encoder.output_len)) 
        return tf.reduce_mean(penalties)  

    def triplet_metrics(self, y_true, y_pred):
        return self.triplet_loss(y_true, y_pred)
    
    def homopolymer_metrics(self, y_true, y_pred):
        return self.homopolymer_loss(y_true, y_pred)
    
    def gc_content_metrics(self, y_true, y_pred):
        return self.gc_balance_loss(y_true, y_pred) 
    
    def pairwise_distance(self, features):
        anchor = features*tf.one_hot(tf.math.argmax(features,2),4)
        pairwise_distances = tf.reduce_sum(tf.abs(anchor[:, None, :, :] - anchor[None, :, :, :]),axis=[2,3])/(2*self.encoder.output_len)
        num_data = tf.shape(anchor)[0]
        mask_diagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(tf.ones(num_data))
        pairwise_distances = tf.math.multiply(pairwise_distances, mask_diagonals)
        return pairwise_distances
