#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import os
import numpy as np
import tensorflow as tf
# import tensorflow_addons.losses as tfa_loss
import pandas as pd
import argparse
from models.encoder import DNAEncoder
from models.triplet_network import TripletNetwork
from models.triplet_dataset import TripletDataset
import matplotlib.pyplot as plt
from models.seqtools import onehots_to_seqs, seqs_to_onehots

def encode_queries(encoder, train_data, num_classes):
    train_features = pd.read_hdf(train_data)
    query_seqs = []

    for i in tqdm(range(num_classes), desc="Encoding queries"):
        group_dna = train_features[train_features.index.str.endswith(f"_{i}")]
        train_seqs = encoder.encode_feature_seqs(group_dna)
        dna_seqs = seqs_to_onehots(train_seqs).sum(0).reshape(1, -1, 4)
        query_seqs.append(onehots_to_seqs(dna_seqs))
    return query_seqs

def setup_datasets(path):
    data = pd.read_hdf(path)
    features = data.values
    labels = np.array([int(tbl_index.split("_")[-1]) for tbl_index in data.index])
    return features, labels

def main():
    # Set GPU memory growth
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    parse = argparse.ArgumentParser()
    parse.add_argument("--train_data", type=str, help="Path to the training data")
    parse.add_argument("--test_data", type=str, help="Path to the test data")
    parse.add_argument("-n", "--num_classes", type=int, help="Number of classes", default=10)
    parse.add_argument("--target_seqs", type=str, help="Output path for the target sequences")
    parse.add_argument("--query_seqs", type=str, help="Output path for the query sequences")
    parse.add_argument("--encoder_path", type=str, help="Path to the encoder")

    args = parse.parse_args()
    
    train_features, train_labels = setup_datasets(args.train_data)
    test_features, test_labels = setup_datasets(args.test_data)
    encoder = DNAEncoder()
    network = TripletNetwork(encoder, num_classes=args.num_classes)

    train_dataset = TripletDataset(encoder, train_features, train_labels)
    val_dataset = TripletDataset(encoder, test_features, test_labels)

    history = network.model.fit(
        train_dataset.dataset,
        validation_data = val_dataset.dataset,
        epochs = 150,
        verbose = 1
    )
    # Save the encoder
    encoder.save(args.encoder_path)
    # Plot the training loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    save_path = args.encoder_path.split(".")[0]+".png"
    plt.savefig(save_path)

    os.makedirs(os.path.dirname(args.encoder_path), exist_ok=True)

    print("Encoding the sequences")
    # encode the features into DNA sequences
    target_features = pd.read_hdf(args.test_data)
    target_seqs = encoder.encode_feature_seqs(target_features)
    query_seqs = encode_queries(encoder, args.train_data, args.num_classes)

    print("Saving the sequences")
    # Save the DNA sequences
    pd.DataFrame(query_seqs, index=[i for i in range(args.num_classes)], columns=['FeatureSequence']).to_hdf(args.query_seqs, key='df', mode='w')
    pd.DataFrame(target_seqs, index=target_features.index, columns=['FeatureSequence']).to_hdf(args.target_seqs, key='df', mode='w')
    


if __name__=="__main__":
    main()
