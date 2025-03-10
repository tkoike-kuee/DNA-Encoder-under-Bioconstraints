import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
import random
from .dataset import Dataset
from tqdm import tqdm

class OpenImagesTrain(Dataset):
    def __init__(self, path):
        self.path = path
 
    def get_images(self, img_ids):
        image_dir = os.path.join(self.path, 'images')

        for img_id in img_ids:
            img_path = os.path.join(
                image_dir, '%s/%s.jpg' % (img_id[:2], img_id)
            )
            image = Image.open(img_path)
            image.load()
            yield image

    def random_pairs(self, batch_size):
        df = pd.read_hdf(self.path)
        n = len(df)
        while True:
            pairs = np.random.permutation(n)[:batch_size*2].reshape(-1,2)

            yield df.index.values[pairs], df.values[pairs]

    def makeDataset(self, batch_size):
        df = pd.read_hdf(self.path)

        x=df.values
        y=np.array([int(tbl_index.split("_")[-1]) for tbl_index in df.index]).reshape(-1,1)
        return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=len(x)*2).batch(batch_size,drop_remainder=True)


    def makePairs(self, num_classes=10):
        df = pd.read_hdf(self.path)

        x = df.values
        y = np.array([int(tbl_index.split("_")[-1]) for tbl_index in df.index])

        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

        pairs = list()
        labels = list()

        for idx1 in range(len(x)):
            # same class
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]
            
            labels += list([1])
            pairs += [[x1, x2]]
        
            # different class
            label2 = random.randint(0, num_classes-1)
            while label2 == label1:
                label2 = random.randint(0, num_classes-1)

            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]
            
            labels += list([0])
            pairs += [[x1, x2]]
            
        return np.array(pairs), np.array(labels)
                
class OpenImagesVal(Dataset):
    
    def __init__(self, val_path):
        self.feature_path = os.path.join(val_path)
        self.df = pd.read_hdf(self.feature_path)
        print(self.df)
        
    def random_pairs(self, batch_size):
        n = len(self.df)
        while True:
            pairs = np.random.permutation(n)[:batch_size*2].reshape(-1,2)
            yield self.df.index.values[pairs], self.df.values[pairs]
    
    def makeDataset(self,batch_size):
        x=self.df.values
        y=np.array([int(tbl_index.split("_")[-1]) for tbl_index in self.df.index]).reshape(-1,1)
        return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=len(x)*2).batch(batch_size,drop_remainder=True)

    def makePairs(self, num_classes=10):
        x = self.df.values
        y = np.array([int(tbl_index.split("_")[-1]) for tbl_index in self.df.index])

        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

        pairs = list()
        labels = list()

        for idx1 in range(len(x)):
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]
            
            labels += list([1])
            pairs += [[x1, x2]]
        
            label2 = random.randint(0, num_classes-1)
            while label2 == label1:
                label2 = random.randint(0, num_classes-1)

            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]
            
            labels += list([0])
            pairs += [[x1, x2]]
            
        return np.array(pairs), np.array(labels)