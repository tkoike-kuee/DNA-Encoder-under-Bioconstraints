from .seqtools import onehots_to_seqs
import tensorflow as tf
import numpy as np
import re

class HPCount(tf.keras.callbacks.Callback):
    def __init__(self, x_val, max_homopolymer, **kwargs):
        super().__init__(**kwargs)
        self.x_val = x_val
        self.max_homopolymer = max_homopolymer
        self.hpoly_counts = {}

    def on_epoch_begin(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val)
        seqs = onehots_to_seqs(y_pred)
        self.hpoly_counts = self.count_homopolymer(seqs)
        tf.print(f"Epoch {epoch + 1}: \nHomopolymer counts: {sorted(self.hpoly_counts.items())}")

        gc_prob = tf.reduce_sum(y_pred[:, :, 0:2], axis=-1)  # (batch, seq_len)
        gc_prob = tf.reduce_sum(gc_prob, axis=-1)  # (batch,)
        tf.print(f"GC contents: {tf.reduce_mean(gc_prob/self.model.output_shape[1])}")


    def count_homopolymer(self, dna_sequence):
        # Initialize dictionary for counting homopolymer lengths
        homopolymer_counts = {}

        # Define regex patterns for homopolymers of A, T, C, and G
        patterns = {base: re.compile(f"{base}{{2,}}") for base in "ATCG"}
            # Search for homopolymers
        for seq in dna_sequence:
            for base, pattern in patterns.items():
                for match in pattern.finditer(seq):
                    homopolymer_length = len(match.group())
                    if homopolymer_length not in homopolymer_counts:
                        homopolymer_counts[homopolymer_length] = 0
                    homopolymer_counts[homopolymer_length] += 1
    
        return homopolymer_counts
