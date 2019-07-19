import numpy as np
import torch
from tqdm import tqdm


PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class Vocabulary(object):
    def __init__(self, special_tokens):
        self.stoi = {}
        self.itos = {}
        self.stof = {}
        self.size = 0
        for token in special_tokens:
            self.add_word(token)

    def __len__(self):
        return self.size

    def has(self, word):
        return word in self.stoi

    def add_word(self, word):
        if not self.has(word):
            ind = self.size
            self.stoi[word] = ind
            self.itos[ind] = word
            self.size += 1

    def to_idx(self, word):
        if self.has(word):
            return self.stoi[word]
        else:
            return self.stoi[UNK_WORD]

    def to_word(self, ind):
        if ind >= self.size:
            return 0
        return self.itos[ind]

    def build_from_counter(self, counter, max_vocab_size=None):
        if max_vocab_size:
            max_vocab_size = max_vocab_size - self.size
        else:
            max_vocab_size = len(counter) + self.size
        sorted_list = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for token, freq in sorted_list[:max_vocab_size]:
            self.add_word(token)
            self.stof[token] = freq


def get_pretrained_embedding(stoi, pretrained_w2v_path, init="random", ret="tensor"):
    mu, sigma = 0, 0.01
    hit = 0
    with open(pretrained_w2v_path, "r") as f:
        line = f.readline()
        word_num, vec_dim = line.split(" ")
        vec_dim = int(vec_dim)
        if init == "random":
            res_embed_matrix = np.array([np.random.normal(mu, sigma, vec_dim).tolist()
                                         for _ in range(len(stoi))])
        elif init == "zero":
            res_embed_matrix = np.zeros((len(stoi), vec_dim))
        else:
            raise NotImplementedError
        print("Total pretrained word num: ", word_num)
        for i, line in tqdm(enumerate(f)):
            word = line.split(" ")[0]
            vec = [float(t) for t in line.strip().split(" ")[1:]]
            if len(vec) != vec_dim:
                # print("\nLine %d: parsed vec(%dd) not match vec %dd" \
                #                         % (i, len(vec), vec_dim))
                continue
            if word in stoi:
                hit += 1
                res_embed_matrix[stoi[word]] = vec
        print("Hit: {}/{}".format(hit, len(stoi)))
    if ret == "tensor":
        res_embed_tensor = torch.tensor(res_embed_matrix)
        return res_embed_tensor
    elif ret == "ndarray":
        return res_embed_matrix
    elif ret == "list":
        return res_embed_matrix.tolist()

