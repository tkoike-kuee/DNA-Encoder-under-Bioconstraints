import numpy as np

bases = np.array(list("ATCG"))
randseq = lambda n: "".join(np.random.choice(bases, n))
complement = dict(zip("ATCG", "TAGC"))
revcomp = lambda s: "".join(reversed([complement[b] for b in s]))

def onehots_to_seqs(onehots):
    return np.array([
        "".join(seq) for seq in bases[onehots.argmax(-1)]
    ])

def seqs_to_onehots(seqs):
    seq_array = np.array(list(map(list, seqs)))
    return np.array([(seq_array == b).T for b in bases]).T.astype(int)

