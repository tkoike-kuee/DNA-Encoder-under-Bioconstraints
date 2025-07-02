#!/usr/bin/env python
# coding: utf-8

from models.encoder import DNAEncoder
from models.triplet_network import TripletNetwork
from models.triplet_dataset import TripletDataset
from models.callback_hpcount import HPCount
from models.seqtools import onehots_to_seqs, seqs_to_onehots
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import argparse

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
    parse.add_argument("--hp", type=int, help="Number of homopolymer", default=4)
    parse.add_argument("--epoch", type=int, help="Number of epochs", default=1000)
    parse.add_argument("--margin", type=float, help="Margin", default=0.8)
    parse.add_argument("--hp_loss_flag", help="Homopolymer loss flag", action='store_false')
    parse.add_argument("--gc_loss_flag", help="GC loss flag", action='store_false')
    parse.add_argument("--encode-only", help="Encode only", action='store_true')

    args = parse.parse_args()
    
    train_features, train_labels = setup_datasets(args.train_data)
    test_features, test_labels = setup_datasets(args.test_data)
    if os.path.isfile(args.encoder_path) or os.path.isdir(args.encoder_path):
        encoder = DNAEncoder(model_path=args.encoder_path, hp=args.hp)
        basename = os.path.basename(args.encoder_path)
        dirname = os.path.dirname(args.encoder_path)
        encoder_path = os.path.join(dirname,"re_"+basename)
    else:
        encoder = DNAEncoder(hp=args.hp)
        encoder_path = args.encoder_path
    if not args.encode_only:
        network = TripletNetwork(encoder, num_classes=args.num_classes, homopolymer=args.hp, hp_loss_flag=int(args.hp_loss_flag), gc_loss_flag=int(args.gc_loss_flag))
        hp_count = HPCount(x_val=test_features, max_homopolymer=args.hp)
        es = tf.keras.callbacks.EarlyStopping(monitor='triplet_metrics', patience=10, min_delta=1e-4)

        train_dataset = TripletDataset(encoder, train_features, train_labels)
        val_dataset = TripletDataset(encoder, test_features, test_labels)

        history = network.model.fit(
            train_dataset.dataset,
            validation_data = val_dataset.dataset,
            epochs = args.epoch,
            verbose = 1,
            callbacks=[hp_count, es],
        )
        # Save the encoder
        encoder.save(encoder_path)
        #Plot the training loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        save_path = args.encoder_path.split(".")[0]+".png"
        plt.savefig(save_path)
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(save_path.replace(".png", ".csv"), index=False)

        os.makedirs(os.path.dirname(args.encoder_path), exist_ok=True)

    print("Encoding the sequences")
    # encode the features into DNA sequences
    target_features = pd.read_hdf(args.test_data)
    target_seqs = encoder.encode_feature_seqs(target_features)
    query_seqs = encode_queries(encoder, args.train_data, args.num_classes)

    print("Saving the sequences")
    # Save query sequences to a FASTA file
    query_records = [
        SeqRecord(Seq(seq[0]), id=str(i), description="") for i, seq in enumerate(query_seqs)
    ]
    SeqIO.write(query_records, args.query_seqs, "fasta")

    # Save target sequences to a FASTA file
    target_records = [
        SeqRecord(Seq(seq), id=str(idx), description="") for idx, seq in zip(target_features.index, target_seqs)
    ]
    SeqIO.write(target_records, args.target_seqs, "fasta")
    


if __name__=="__main__":
    main()
