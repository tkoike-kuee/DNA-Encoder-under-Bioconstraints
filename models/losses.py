import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

class Losses():
  def __init__(self, margin=1.0, seq_len=80, batch=128):
    self.margin=margin
    self.seq_len=seq_len
    self.batch=batch

  def contrastive_loss(self, y_true, y_pred):
    return y_true*tf.math.square(y_pred)+(1.0-y_true)*tf.math.square(tf.math.maximum(self.margin-y_pred, 0.0))
  
  def triplet_loss(self, y_true, y_pred): #semi-hard-loss
    positive_pred, negative_pred = y_pred
    return tf.math.maximum(tf.math.square(positive_pred)-tf.math.square(negative_pred)+self.margin, 0)

  def HammingDistance(self, x, y):
    dist=tf.math.reduce_sum(tf.math.abs(tf.subtract(x,y)),[1,2])/(2*self.seq_len)
    return dist

  def make_dist_mat(self, dna_seq, batch):
    # Create a distance matrix for the batch
    # dna_seq: tensor of shape (batch, seq_len, 4)
    # batch: batch size

    i_vec = tf.range(0, batch) # vector of indices from 0 to batch-1
    j_vec = tf.range(0, batch) # vector of indices from 0 to batch-1
    pairs = tf.stack(tf.meshgrid(i_vec, j_vec, indexing="ij"), axis=-1) # tensor of all pairs of indices
    pairs = tf.reshape(pairs, (-1, 2)) # tensor of all pairs of indices 
  
    mask=tf.math.greater(pairs[:, 0], pairs[:, 1]) # mask to remove duplicate pairs
    indices = tf.boolean_mask(pairs, mask) # tensor of all pairs of indices without duplicates
    dna_i = tf.gather(dna_seq, indices[:,0])  # tensor of DNA sequences corresponding to i
    dna_j = tf.gather(dna_seq, indices[:,1])  # tensor of DNA sequences corresponding to j
    distance = self.HammingDistance(dna_i, dna_j) # tensor of Hamming distances between i and j
    dist_mat = tf.zeros((batch*batch,)) # initialize distance matrix
    dist_mat = tf.tensor_scatter_nd_update(dist_mat, tf.expand_dims(indices[:,0]*batch+indices[:,1], axis=1), distance) # update distance matrix
    dist_mat = tf.reshape(dist_mat, (batch, batch)) # reshape distance matrix
    dist_mat += tf.linalg.band_part(dist_mat, 0, -1) - tf.linalg.band_part(dist_mat, 0, 0) # make distance matrix symmetric
    return dist_mat


  def make_triplet(self, dna_seq):
    # Create a triplet for the batch
    batch=self.batch
    embeddings = tf.convert_to_tensor(dna_seq, name="embeddings")
    if embeddings.shape[0] is not None:
      batch = embeddings.shape[0]
    pairwise_distance = self.make_dist_mat(embeddings, batch)

    return pairwise_distance
