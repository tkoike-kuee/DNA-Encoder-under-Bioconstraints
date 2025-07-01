import numpy as np
import itertools

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

def seek_homopolymer(args, min_length=5):
    length = 0
    begin = []
    end = []
    i = []
    for k, g in itertools.groupby(args):
        hp = len(list(g))
        length += hp
        if hp > min_length:
            begin.append(length - hp)
            end.append(length)
            i.append(k)
    return begin, end, i

def select_base(hp_prob, begin, min_length=5):
    prob = hp_prob.max(-1)
    hp = prob.shape[0]
    sorted_indices = prob.argsort()
    bases = []
    for i in sorted_indices:
        bases.append(begin + i)
        if hp - i - 1 <= min_length:
            break
    return bases

def encode_to_seqs(prob, min_length=5):
    adjust_prob = prob.copy()
    seqs = prob.argmax(-1)
    for i, args in enumerate(seqs):
        begin, end, k = seek_homopolymer(args, min_length)
        while len(begin) > 0:
            for b, e, j in zip(begin, end, k):
                extract_hp = adjust_prob[i,b:e]
                change_indices = select_base(extract_hp, b, min_length)
                adjust_prob[i, change_indices, j] = prob[i, change_indices, j] * -1
            args = adjust_prob[i].argmax(-1)
            begin, end, k = seek_homopolymer(args, min_length)
    return onehots_to_seqs(adjust_prob)