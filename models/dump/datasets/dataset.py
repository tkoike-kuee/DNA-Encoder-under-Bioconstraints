import abc
import numpy as np

class Dataset(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def random_pairs(self, batch_size):
        pass

    def balanced_pairs(self, batch_size):

        pair_generator = self.random_pairs(batch_size)

        while True:

            n_batch = 0
            batch_ids  = []
            batch_vals = []

            while n_batch < batch_size:
                chunk_ids, chunk_vals = next(pair_generator)
                similar = np.apply_along_axis(lambda x: x[0].split("_")[-1]==x[1].split("_")[-1], 1, chunk_ids)

                n_sim = similar.sum()

                batch_ids.extend([
                    chunk_ids[similar],
                    chunk_ids[~similar][:n_sim]
                ])
                
                batch_vals.extend([
                    chunk_vals[similar],
                    chunk_vals[~similar][:n_sim]
                ])

                n_batch += 2 * n_sim

            batch_ids = np.concatenate(batch_ids)
            batch_vals = np.concatenate(batch_vals)

            perm = np.random.permutation(len(batch_vals))[:batch_size]

            yield batch_ids[perm], batch_vals[perm]


class Static(Dataset):

    def __init__(self, X):
        self.X = X

    def random_pairs(self, batch_size):
        n,d = self.X.shape
        while True:
            pairs = np.random.permutation(n)[:batch_size*2].reshape(-1,2)
            yield pairs, self.X[pairs]



